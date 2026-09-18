# Notices and Third-Party Licenses

This repository is licensed under the **GNU Affero General Public License v3.0**.
The full text is in [LICENSE](./LICENSE).

AGPL-3.0 was chosen because the repository vendors AGPL-3.0 and GPL-3.0 code
directly in-tree (see below). Both are strong copyleft licenses, and AGPL-3.0 is
the more restrictive of the two, so it governs the combined work.

## What AGPL-3.0 means here

- Anyone you distribute this software to may request the complete corresponding
  source code.
- **Network use counts as distribution.** If you run the webhook server, the
  dashboard, or the Ghostfolio app where others can interact with it over a
  network, those users are entitled to the source of the version they are
  interacting with. This is the clause that distinguishes AGPL from GPL, and it
  applies to a hosted trading service.
- Modifications and works based on this code must be released under AGPL-3.0.

Running it privately for yourself triggers none of these obligations.

## Vendored third-party components

| Path | Component | Version | License | Upstream |
|------|-----------|---------|---------|----------|
| `ghostfolio/` | Ghostfolio | 3.70.1 | AGPL-3.0 | https://github.com/ghostfolio/ghostfolio |
| `alpaca-trend-alert/` | alpaca-trend-alert | — | GPL-3.0 | see `alpaca-trend-alert/README.md` |

Each retains its own `LICENSE` file, and those govern the files beneath them.
Neither has been modified from upstream beyond being committed as regular files
(commit `6df3ae8`).

## Python dependencies

Declared in `requirements.txt`, installed from PyPI, each under its own license.
None are redistributed in this repository.

## Note for maintainers

If the copyleft obligations above are not wanted, the alternative is to remove
`ghostfolio/` and `alpaca-trend-alert/` from the tree and reference them as
external dependencies or git submodules instead. No first-party Python module
imports either directory, so nothing in `shared/`, `strategies/`, or the broker
integrations depends on them being present.
