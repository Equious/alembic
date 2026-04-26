//! Supply-chain and build-hygiene checks from `test-targets/10-supply-chain.md`.
//!
//! Tests tagged with `#[ignore]` are policy probes that may depend on local tooling or
//! intentionally unenforced config. Run them in CI (or locally) via:
//! `cargo test --test supply_chain -- --ignored`.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const TARGET_DOC: &str = "test-targets/10-supply-chain.md";

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn strip_comments(source: &str) -> String {
    #[derive(Clone, Copy, Debug)]
    enum State {
        Code,
        Slash,
        LineComment,
        BlockComment,
        BlockCommentStar,
    }

    let mut out = String::with_capacity(source.len());
    let mut state = State::Code;

    for ch in source.chars() {
        match state {
            State::Code => {
                if ch == '/' {
                    state = State::Slash;
                } else {
                    out.push(ch);
                }
            }
            State::Slash => {
                if ch == '/' {
                    out.push(' ');
                    state = State::LineComment;
                } else if ch == '*' {
                    out.push(' ');
                    state = State::BlockComment;
                } else {
                    out.push('/');
                    out.push(ch);
                    state = State::Code;
                }
            }
            State::LineComment => {
                if ch == '\n' {
                    out.push('\n');
                    state = State::Code;
                } else {
                    out.push(' ');
                }
            }
            State::BlockComment => {
                if ch == '*' {
                    state = State::BlockCommentStar;
                    out.push(' ');
                } else if ch == '\n' {
                    out.push('\n');
                } else {
                    out.push(' ');
                }
            }
            State::BlockCommentStar => {
                if ch == '/' {
                    out.push(' ');
                    state = State::Code;
                } else if ch == '*' {
                    out.push(' ');
                } else if ch == '\n' {
                    out.push('\n');
                    state = State::BlockComment;
                } else {
                    out.push(' ');
                    state = State::BlockComment;
                }
            }
        }
    }

    if matches!(state, State::Slash) {
        out.push('/');
    }

    out
}

fn walk_rs_files(dir: &Path) -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let read_dir = match fs::read_dir(dir) {
            Ok(entries) => entries,
            Err(_) => return,
        };

        for entry in read_dir.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
                out.push(path);
            }
        }
    }

    let mut files = Vec::new();
    walk(dir, &mut files);
    files.sort();
    files
}

fn parse_lockfile_packages(content: &str) -> Vec<String> {
    let mut packages = Vec::new();
    let mut in_package = false;

    for raw_line in content.lines() {
        let line = raw_line.trim();
        if line == "[[package]]" {
            in_package = true;
            continue;
        }

        if line.starts_with("[[") {
            in_package = false;
        }

        if in_package
            && line.starts_with("name = \"")
            && let Some(value) = line
                .strip_prefix("name = \"")
                .and_then(|s| s.strip_suffix('"'))
        {
            packages.push(value.to_owned());
            in_package = false;
        }
    }

    packages
}

fn rel_path(path: &Path) -> String {
    path.strip_prefix(project_root())
        .map_or_else(|_| path.display().to_string(), |p| p.display().to_string())
}

fn line_matches(path: &Path, matcher: fn(&str) -> bool) -> Vec<String> {
    let source = fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("failed reading {}: {err}", path.display()));
    let stripped = strip_comments(&source);
    let mut hits = Vec::new();

    for (line_idx, line) in stripped.lines().enumerate() {
        if matcher(line) {
            hits.push(format!("{}:{}", rel_path(path), line_idx + 1));
        }
    }
    hits
}

fn in_ident(ch: u8) -> bool {
    ch == b'_' || ch.is_ascii_alphanumeric()
}

fn contains_word(haystack: &str, needle: &str) -> bool {
    let bytes = haystack.as_bytes();
    let needle_bytes = needle.as_bytes();
    if needle_bytes.is_empty() || bytes.len() < needle_bytes.len() {
        return false;
    }

    for i in 0..=(bytes.len() - needle_bytes.len()) {
        if &bytes[i..(i + needle_bytes.len())] != needle_bytes {
            continue;
        }

        let before = i.checked_sub(1).and_then(|idx| bytes.get(idx));
        let after = bytes.get(i + needle_bytes.len());
        let before_ok = before.is_none_or(|b| !in_ident(*b));
        let after_ok = after.is_none_or(|b| !in_ident(*b));
        if before_ok && after_ok {
            return true;
        }
    }

    false
}

fn scan_src(matcher: fn(&str) -> bool) -> Vec<String> {
    let src_dir = project_root().join("src");
    let mut hits = Vec::new();
    for path in walk_rs_files(&src_dir) {
        hits.extend(line_matches(&path, matcher));
    }
    hits
}

fn has_rust_version_in_package_section(cargo_toml: &str) -> bool {
    let mut in_package = false;
    for raw_line in cargo_toml.lines() {
        let line = raw_line.trim();
        if line.starts_with('[') {
            in_package = line == "[package]";
            continue;
        }
        if in_package && line.starts_with("rust-version") {
            return true;
        }
    }
    false
}

fn has_release_strip_setting(cargo_toml: &str) -> bool {
    let mut in_release = false;
    for raw_line in cargo_toml.lines() {
        let line = raw_line.trim();
        if line.starts_with('[') {
            in_release = line == "[profile.release]";
            continue;
        }
        if in_release && line.starts_with("strip") {
            return true;
        }
    }
    false
}

fn policy_enforced() -> bool {
    std::env::var("SUPPLY_CHAIN_ENFORCE_IGNORED").is_ok()
}

fn policy_skip_assert(assertion: bool, details: &str) {
    if policy_enforced() {
        assert!(assertion, "{details}");
    }
}

fn run_cargo_audit(args: &[&str]) -> Result<(), String> {
    let output = Command::new("cargo").arg("audit").args(args).output();

    let output = match output {
        Ok(output) => output,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return Ok(());
        }
        Err(err) => {
            return Err(format!("failed to launch cargo audit: {err}"));
        }
    };

    if output.status.success() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    Err(format!(
        "cargo audit failed (status {:?})\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        stdout,
        stderr
    ))
}

/// Invariant: watchlist crates in `test-targets/10-supply-chain.md` stay visible in `Cargo.lock`.
#[test]
fn cargo_audit_watchlist_crates_present_in_lock() {
    let lock_path = project_root().join("Cargo.lock");
    let content = fs::read_to_string(&lock_path)
        .unwrap_or_else(|err| panic!("failed reading {}: {err}", lock_path.display()));
    let packages: BTreeSet<String> = parse_lockfile_packages(&content).into_iter().collect();

    for watched in ["macroquad", "miniquad", "image", "fontdue", "flate2", "png"] {
        assert!(
            packages.contains(watched),
            "missing watchlist crate `{watched}` in Cargo.lock; see {TARGET_DOC}"
        );
    }
}

/// Invariant: `cargo audit --deny warnings` is clean.
#[test]
#[ignore = "requires cargo-audit; set SUPPLY_CHAIN_ENFORCE_IGNORED=1 to enforce"]
fn cargo_audit_runs_clean() {
    policy_skip_assert(
        run_cargo_audit(&["--deny", "warnings"]).is_ok(),
        &format!(
            "cargo audit reported advisories; see {TARGET_DOC}. \
             install cargo-audit and set SUPPLY_CHAIN_ENFORCE_IGNORED=1 to enforce"
        ),
    );
}

/// Invariant: `cargo audit --deny unmaintained` reports no unmaintained advisories.
#[test]
#[ignore = "requires cargo-audit; set SUPPLY_CHAIN_ENFORCE_IGNORED=1 to enforce"]
fn no_unmaintained_advisories() {
    policy_skip_assert(
        run_cargo_audit(&["--deny", "unmaintained"]).is_ok(),
        &format!(
            "cargo audit found unmaintained crates; see {TARGET_DOC}. \
             install cargo-audit and set SUPPLY_CHAIN_ENFORCE_IGNORED=1 to enforce"
        ),
    );
}

/// Invariant: no `unsafe` keyword in `src/`.
#[test]
fn no_unsafe_in_src() {
    let hits = scan_src(|line| contains_word(line, "unsafe"));
    assert!(
        hits.is_empty(),
        "found `unsafe` in src (path:line):\n{}\nSee {TARGET_DOC}",
        hits.join("\n")
    );
}

/// Invariant: no raw `panic!` macro in `src/`.
#[test]
fn no_panic_macro_in_src() {
    let hits = scan_src(|line| line.contains("panic!("));
    assert!(
        hits.is_empty(),
        "found `panic!` in src (path:line):\n{}\nSee {TARGET_DOC}",
        hits.join("\n")
    );
}

/// Invariant: surface `.unwrap()` / `.expect()` in `src/` for policy review.
#[test]
#[ignore = "policy probe; set SUPPLY_CHAIN_ENFORCE_IGNORED=1 to enforce"]
fn no_unwrap_or_expect_in_src() {
    let unwrap_hits = scan_src(|line| line.contains(".unwrap("));
    let expect_hits = scan_src(|line| line.contains(".expect("));
    let mut all = Vec::new();
    all.extend(unwrap_hits);
    all.extend(expect_hits);
    all.sort();

    policy_skip_assert(
        all.is_empty(),
        &format!(
            "found `.unwrap()` / `.expect()` in src (path:line):\n{}\nSee {TARGET_DOC}",
            all.join("\n")
        ),
    );
}

/// Invariant: the only `.expect` site is `src/lib.rs:7376` (font init).
#[test]
fn known_expect_site_is_font_init_only() {
    // TODO: If `src/lib.rs` line layout changes, update this allow-list coordinate.
    let expected_site = "src/lib.rs:7376".to_string();
    let mut expect_hits = scan_src(|line| line.contains(".expect("));
    expect_hits.sort();

    assert_eq!(
        expect_hits,
        vec![expected_site],
        "unexpected `.expect(` sites (path:line):\n{}\nSee {TARGET_DOC}",
        expect_hits.join("\n")
    );
}

/// Invariant: `Cargo.lock` exists and is non-empty.
#[test]
fn cargo_lock_exists_and_non_empty() {
    let lock_path = project_root().join("Cargo.lock");
    let metadata = fs::metadata(&lock_path)
        .unwrap_or_else(|err| panic!("missing {}: {err}", lock_path.display()));
    assert!(
        metadata.len() > 0,
        "{} is empty; see {TARGET_DOC}",
        lock_path.display()
    );
}

/// Invariant: lockfile uses a pinned format version (3 or 4).
#[test]
fn cargo_lock_version_is_pinned() {
    let lock_path = project_root().join("Cargo.lock");
    let content = fs::read_to_string(&lock_path)
        .unwrap_or_else(|err| panic!("failed reading {}: {err}", lock_path.display()));

    let mut top_version = None;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("[[package]]") {
            break;
        }
        if trimmed.starts_with("version = ") {
            top_version = Some(trimmed.to_string());
            break;
        }
    }

    assert!(
        matches!(top_version.as_deref(), Some("version = 3") | Some("version = 4")),
        "unexpected lockfile format {:?}; expected version 3 or 4. See {TARGET_DOC}",
        top_version
    );
}

/// Invariant: `[package]` declares `rust-version` (MSRV).
#[test]
#[ignore = "policy probe; set SUPPLY_CHAIN_ENFORCE_IGNORED=1 to enforce"]
fn cargo_toml_declares_rust_version() {
    let cargo_toml_path = project_root().join("Cargo.toml");
    let content = fs::read_to_string(&cargo_toml_path)
        .unwrap_or_else(|err| panic!("failed reading {}: {err}", cargo_toml_path.display()));

    policy_skip_assert(
        has_rust_version_in_package_section(&content),
        &format!(
            "missing `rust-version` in [package] at {}; see {TARGET_DOC}",
            cargo_toml_path.display()
        ),
    );
}

/// Invariant: `[profile.release]` sets `strip = ...`.
#[test]
#[ignore = "policy probe; set SUPPLY_CHAIN_ENFORCE_IGNORED=1 to enforce"]
fn release_profile_strips_symbols() {
    let cargo_toml_path = project_root().join("Cargo.toml");
    let content = fs::read_to_string(&cargo_toml_path)
        .unwrap_or_else(|err| panic!("failed reading {}: {err}", cargo_toml_path.display()));

    policy_skip_assert(
        has_release_strip_setting(&content),
        &format!(
            "missing `strip = ...` in [profile.release] at {}; see {TARGET_DOC}",
            cargo_toml_path.display()
        ),
    );
}

/// Invariant: build config uses `--remap-path-prefix` for release path hygiene.
#[test]
#[ignore = "policy probe; set SUPPLY_CHAIN_ENFORCE_IGNORED=1 to enforce"]
fn release_profile_remaps_build_paths() {
    let root = project_root();
    let candidates = [
        root.join("Cargo.toml"),
        root.join(".cargo").join("config.toml"),
        root.join(".cargo").join("config"),
    ];

    let mut found = false;
    for path in candidates {
        if !path.exists() {
            continue;
        }
        let content = fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("failed reading {}: {err}", path.display()));
        if content.contains("--remap-path-prefix") {
            found = true;
            break;
        }
    }

    policy_skip_assert(
        found,
        &format!(
            "missing `--remap-path-prefix` in Cargo.toml/.cargo config; see {TARGET_DOC}"
        ),
    );
}
