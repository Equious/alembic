# 10 — Supply Chain & Build Hygiene

## Scope

Non-simulation targets that benefit from an automated agent sweep. These
are the natural lane for a security-focused tool like Cygent and require
no headless harness.

## Prerequisites

None. These targets run against `Cargo.toml`, `Cargo.lock`, and the source
tree directly.

## Targets

### RustSec advisories

- **`cargo audit` is clean.** No known vulnerabilities in any crate in
  `Cargo.lock`. Current transitive deps worth specifically watching:
  `macroquad`, `miniquad`, `image`, `fontdue`, `flate2`, `png`.

  *How Cygent can help:* schedule `cargo audit` on every PR, open an
  issue (or fail the PR) when a new advisory lands.

### Unmaintained crates

- **No unmaintained-flagged deps in `Cargo.lock`.** `cargo audit` also
  reports unmaintained crates. Action: keep the list empty, or document
  exceptions with a reason here.

### Unsafe scan

- **Zero `unsafe` blocks in `src/`.** Scan our own code (not transitive
  dependencies) and assert no `unsafe` keyword appears.

  *Why:* The sim is pure-safe Rust; any introduction of `unsafe` should
  be deliberate and reviewed. Transitive deps inevitably use `unsafe`
  (`macroquad`, `miniquad`, GPU interop) — that's out of scope.

### Panic-on-release hygiene

- **No raw `panic!` macros in sim code paths.** Audit `src/` for
  `panic!(...)` calls reachable from the tick loop. Asserts and
  `unreachable!()` in truly unreachable branches are acceptable; raw
  `panic!` on reachable paths is not.

- **No `.unwrap()` / `.expect()` in `World::step` or its descendants.**
  The call graph rooted at `step` must not contain `.unwrap()` on
  anything that could fail under user input or fuzz. Getters that are
  provably infallible (e.g., `Vec::get(x).unwrap()` where `x` was just
  bounds-checked) are acceptable with a comment, but cleaner to use
  `.expect("reason")` or refactor to propagate `Option`.

### Dependency drift

- **`Cargo.lock` matches CI environment.** No accidental bumps from
  `cargo update` slipping into a PR without review. The lock file is
  canonical; check it in, don't regenerate silently.

- **MSRV documented.** Pin the minimum supported Rust version in
  `Cargo.toml` (`rust-version = "..."`) so CI and Steam build machines
  agree on toolchain.

### Binary hygiene (release build)

- **Release binary doesn't link debug symbols unexpectedly.** Strip or
  `strip = "symbols"` in release profile (optional — include if we
  decide we want smaller binaries).

- **Release binary doesn't embed build paths.** Use `--remap-path-prefix`
  or the equivalent so `/home/equious/alembic/...` doesn't leak into
  the shipped binary.

## Known regressions

- (Placeholder — add entries when a supply-chain issue is caught and
  fixed.)

## Out of scope

- Upstream crate code quality — we don't audit macroquad's internals;
  we only track advisories and maintenance status.
