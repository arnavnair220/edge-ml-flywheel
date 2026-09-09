"""One image's content digest and the facts the integrity checks read.

Design section 4.1's data gate wants three things caught before an image is ever
trained on: files that do not decode, files that are not the resolution the rest
of the pipeline assumes, and frames that carry no picture at all. All three are
answered here, and none of them is answered *here* -- this returns what was
measured and `manifest` decides what counts as a finding. Same rule the manifest
schema follows: state the fact, leave the verdict to whoever owns the threshold.

**Nothing in this module raises for a bad image.** An undecodable file still has
a sha256, and a row whose digest is recorded is a row the report can point at.
Raising instead would lose the one piece of evidence about the file that
provoked the failure.

**The decode is drafted down.** A JPEG can be decoded at 1/8 scale straight out
of the DCT coefficients, which is a few times cheaper than a full decode and
answers "is this frame blank" identically -- an all-black image is all-black at
any scale. Across 80,000 images on 2 vCPU that is the difference between a step
measured in minutes and one measured in tens of them. The dimensions are read
before drafting, because drafting is what changes them.
"""

import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Final, cast

from PIL import Image

# 1/8 of `conventions.NATIVE_IMAGE_SIZE`, which is the largest reduction libjpeg
# offers. Not derived from it arithmetically: this is an argument to `draft`,
# which snaps to its own 1/2, 1/4, 1/8 ladder, so a computed value would look
# like a knob that follows the native size when it is really one of three
# choices.
_DRAFT_SIZE: Final = (160, 90)


@dataclass(frozen=True, slots=True)
class ImageFacts:
    """Measurements, not verdicts.

    `size` and `extrema` are `None` exactly when `decode_error` is set, so a
    reader that checks one has checked the other.
    """

    sha256: str
    size: tuple[int, int] | None
    extrema: tuple[int, int] | None
    decode_error: str | None

    @property
    def is_blank(self) -> bool:
        """One luminance value across the whole frame: all-black, all-white, or
        a uniform grey that is neither but is equally not a photograph."""
        return self.extrema is not None and self.extrema[0] == self.extrema[1]


def inspect_image(path: Path) -> ImageFacts:
    """Digest and decode one image. Only an unreadable *file* raises."""
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()

    # Deliberately broad. Pillow answers a corrupt file with any of
    # UnidentifiedImageError, OSError, ValueError or SyntaxError depending on
    # where the bytes go wrong, and the caller's response is the same for all of
    # them: record the file and keep going, so the report names every bad image
    # rather than the first one.
    try:
        with Image.open(BytesIO(data)) as image:
            size = image.size
            image.draft("L", _DRAFT_SIZE)
            # `getextrema` is typed for the multi-band case, where it returns
            # one pair per band. `convert("L")` guarantees a single band, so
            # what comes back here is one pair.
            low, high = cast(tuple[float, float], image.convert("L").getextrema())
    except Exception as error:
        return ImageFacts(sha256=digest, size=None, extrema=None, decode_error=repr(error))

    return ImageFacts(
        sha256=digest,
        size=size,
        extrema=(int(low), int(high)),
        decode_error=None,
    )
