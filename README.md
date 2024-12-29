[![Build Status (nightly)](https://github.com/sigurd4/slice_math/workflows/Build-nightly/badge.svg)](https://github.com/sigurd4/slice_math/actions/workflows/build-nightly.yml)
[![Build Status (nightly, all features)](https://github.com/sigurd4/slice_math/workflows/Build-nightly-all-features/badge.svg)](https://github.com/sigurd4/slice_math/actions/workflows/build-nightly-all-features.yml)

[![Build Status (stable)](https://github.com/sigurd4/slice_math/workflows/Build-stable/badge.svg)](https://github.com/sigurd4/slice_math/actions/workflows/build-stable.yml)
[![Build Status (stable, all features)](https://github.com/sigurd4/slice_math/workflows/Build-stable-all-features/badge.svg)](https://github.com/sigurd4/slice_math/actions/workflows/build-stable-all-features.yml)

[![Test Status](https://github.com/sigurd4/slice_math/workflows/Test/badge.svg)](https://github.com/sigurd4/slice_math/actions/workflows/test.yml)
[![Lint Status](https://github.com/sigurd4/slice_math/workflows/Lint/badge.svg)](https://github.com/sigurd4/slice_math/actions/workflows/lint.yml)

[![Latest Version](https://img.shields.io/crates/v/slice_math.svg)](https://crates.io/crates/slice_math)
[![License:MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/docsrs/slice_math)](https://docs.rs/slice_math)
[![Coverage Status](https://img.shields.io/codecov/c/github/sigurd4/slice_math)](https://app.codecov.io/github/sigurd4/slice_math)

# slice_math

Provides many useful match utility methods for arrays.

Mostly for signal-processing related stuff since that's what i use the most. If anyone has any suggestions for more useful slice math operations, please post them on the github as an issue.

This is one of those crates where i'll just add more stuff when i need it elsewhere, and it will probably never be complete.

Preferrably, i want to use as few external dependencies as possible, other than [`num`](https://crates.io/crates/num).

Temporarily, i'm also using [`ndarray`](https://crates.io/crates/ndarray), but i want to rely on it less than i do now.