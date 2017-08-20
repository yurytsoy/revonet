## [0.2.0] - 2017-08-20

### Added
* Support for neural networks with skip connections.
* Support for performing and analysis of multiple runs for GA and NE.
* Associated types for the EA algorithms for code simplification.
* More documentation examples.
* Keywords and categories for crate description in the toml file.
* Several utility math functions to work with vectors.
* `Jsonable` trait to avoid json import and export code duplication.

### Changed
* Now EA returns `&EAResult` not `Rc<&EAResult>`.
* Now EA has global `Problem` generic instead of local one.
* Optimized computation of NN output: removed generation of extra vectors with output signals.
* Replaced unwrap with expect.

### Removed
* Removed generics for `Individual` type for the GA and NE.
* Duplicated `clone` from `EAIndividual`.
* Unused lifetimes.

### Fixed