#!/bin/bash
# What turns a bare Graviton instance into a Greengrass device. Runs once, at
# first boot, as root.
#
# **Provisioning is manual, in Greengrass's sense of the word.** The installer
# can create its own thing, certificate and policy given AWS credentials, and
# that path is deliberately not taken: it would put resources outside Terraform
# state that nothing here could then destroy, and it needs an instance profile
# that may create IoT identities. Instead Terraform creates all of it and this
# script installs against a config file naming what already exists, with
# `--provision false`.
#
# **The certificate arrives through Parameter Store, not through this file.**
# User data is readable by anything on the instance through IMDS and by anyone
# holding `ec2:DescribeInstanceAttribute`, so a private key pasted here would be
# a private key in two places neither of which is a secret store.
#
# Every value below is rendered by `templatefile` in `fleet.tf`. Nothing is
# discovered at boot: an instance that cannot reach an endpoint should fail here,
# at first boot, rather than on the first deployment.
set -euo pipefail

exec > >(tee /var/log/edge-ml-flywheel-setup.log) 2>&1
echo "provisioning ${thing_name} in ${region}"

dnf -y update
dnf -y install java-17-amazon-corretto-headless python3.12 python3.12-pip unzip

# --- The interpreter the replay component runs under -------------------------
#
# A virtualenv at a path the recipe names as a literal, which is the one path in
# this project a Greengrass recipe variable does not supply -- the nucleus knows
# where it put an artifact and nothing about what the instance built.
#
# World-readable because the component runs as ggc_user and this is built by
# root. Not world-writable: a component that could rewrite its own interpreter
# would make the artifact digests the canary checks say nothing about what ran.
cat > /tmp/requirements.txt <<'REQUIREMENTS'
${requirements}
REQUIREMENTS

python3.12 -m venv "${venv}"
"${venv}/bin/pip" install --no-cache-dir --upgrade pip
"${venv}/bin/pip" install --no-cache-dir -r /tmp/requirements.txt
chmod -R a+rX "$(dirname "${venv}")"

# --- The device's identity ---------------------------------------------------
#
# Written to the Greengrass root with the permissions the nucleus expects. The
# private key is read once, here, and never lands anywhere a later process can
# find it -- `--with-decryption` is what the instance profile is granted, and
# the instance profile is granted nothing else over Parameter Store.
mkdir -p "${greengrass_root}"
aws ssm get-parameter --region "${region}" --name "${certificate_parameter}" \
  --query Parameter.Value --output text > "${greengrass_root}/device.pem.crt"
aws ssm get-parameter --region "${region}" --name "${private_key_parameter}" \
  --with-decryption --query Parameter.Value --output text > "${greengrass_root}/private.pem.key"
chmod 600 "${greengrass_root}/private.pem.key"
chmod 644 "${greengrass_root}/device.pem.crt"

curl -fsSL https://www.amazontrust.com/repository/AmazonRootCA1.pem \
  -o "${greengrass_root}/AmazonRootCA1.pem"

# --- The nucleus -------------------------------------------------------------
#
# The config file names the role alias rather than any credential. That alias is
# how a component gets AWS access at all: the nucleus exchanges the certificate
# above for short-lived credentials from the token exchange role, and there is no
# long-lived key on this device to find.
cat > /tmp/config.yaml <<CONFIG
---
system:
  certificateFilePath: "${greengrass_root}/device.pem.crt"
  privateKeyPath: "${greengrass_root}/private.pem.key"
  rootCaPath: "${greengrass_root}/AmazonRootCA1.pem"
  rootpath: "${greengrass_root}"
  thingName: "${thing_name}"
services:
  aws.greengrass.Nucleus:
    componentType: "NUCLEUS"
    version: "${nucleus_version}"
    configuration:
      awsRegion: "${region}"
      iotRoleAlias: "${role_alias}"
      iotDataEndpoint: "${data_endpoint}"
      iotCredEndpoint: "${credentials_endpoint}"
CONFIG

curl -fsSL "${nucleus_url}" -o /tmp/greengrass-nucleus.zip
unzip -q /tmp/greengrass-nucleus.zip -d /tmp/GreengrassInstaller

# `--setup-system-service true` makes the nucleus a systemd unit, so the device
# comes back up running the component it was last deployed rather than needing
# someone to start it. That is the property a rollback depends on: a restored
# version has to survive whatever made the device restart.
java -Droot="${greengrass_root}" -Dlog.store=FILE \
  -jar /tmp/GreengrassInstaller/lib/Greengrass.jar \
  --init-config /tmp/config.yaml \
  --component-default-user ggc_user:ggc_group \
  --provision false \
  --setup-system-service true

rm -rf /tmp/GreengrassInstaller /tmp/greengrass-nucleus.zip /tmp/config.yaml
echo "nucleus installed, waiting for a deployment"
