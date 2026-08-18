"""What we download, and what we expect it to be.

Two files from one host. `bdd-data.berkeley.edu/download.html`'s buttons are
plain links to a raw UC Berkeley IP, and `dl.yf.io` resolves to that same
address serving byte-identical files. No login, no presigned URLs, no API.

**Plain HTTP and only one published checksum**, so nothing about the transfer is
trustworthy and verification has to happen after it. That is the whole reason
`raw/` is content-addressed: the labels digest below establishes one archive,
and the image-ID set check in the verification suite establishes the other,
which is the archive that has no digest to check.

**Two archives on this host are not what their names say**, which is why the
choice of these two is recorded rather than assumed. `bdd100k_det_20_labels.zip`
contains 2,000 tracking JPEGs and zero labels, and the download page's
`2021/bdd100k_det_20_labels_trainval.zip` link 404s. The legacy 2018 Scalabel
archive named here is a complete substitute -- it carries both the boxes and the
weather/scene/timeofday attributes the wave design runs on -- and is arguably
the better one: `det_20` has a documented gap where train yields 69,863 of
70,000 images with labels, while every image in this archive has detection
boxes.

The archive is licensed for non-commercial research use by the UC Regents, and
the terms are accepted on the portal page rather than implied by the buttons not
enforcing them. `provenance` records that alongside the bytes.
"""

from dataclasses import dataclass
from typing import Final

# The one path component both archives sit under on the host.
_ARCHIVE_PATH: Final = "bdd100k"


@dataclass(frozen=True, slots=True)
class Archive:
    """One downloadable file and the two things we can check it against.

    `sha256` is `None` for the images archive because none is published for it.
    Modelling that as an honest `None` rather than omitting the field is what
    keeps `verify_archive` from quietly checking nothing: the absence is a
    property of the source, and the size check is what stands in for it until
    the ID-set check runs.
    """

    name: str
    size_bytes: int
    sha256: str | None

    def url(self, host: str) -> str:
        return f"http://{host}/{_ARCHIVE_PATH}/{self.name}"


IMAGES: Final = Archive(
    name="bdd100k_images_100k.zip",
    size_bytes=5_669_071_832,
    sha256=None,
)

LABELS: Final = Archive(
    name="bdd100k_labels.zip",
    size_bytes=189_638_612,
    sha256="7f1f9043c70a6ff0788a323cfb914aeced109a37b7150740d670a347c881394a",
)

# Keyed by the word the buildspec passes on the command line.
ARCHIVES: Final[dict[str, Archive]] = {"images": IMAGES, "labels": LABELS}

LICENSE: Final = {
    "name": "BDD100K, non-commercial research use",
    "holder": "The Regents of the University of California",
    "terms_url": "https://bdd-data.berkeley.edu/download.html",
    "note": (
        "Accepted on the portal page. The download buttons do not enforce the terms, "
        "which does not make them optional."
    ),
}
