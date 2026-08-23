# DOI-minting repository release plan

The planned public release will use a trusted DOI-minting general repository such as Zenodo, subject to the corresponding author's final institutional and licensing approval.

## Release contents

- versioned source code corresponding to the accepted manuscript;
- `environment.yml` and `requirements.txt`;
- compact source data underlying every main and supplementary figure;
- data dictionary, figure-to-code map and pipeline description;
- machine-readable file manifest and SHA-256 checksums;
- repository metadata and a software licence selected by the authors;
- links to the published Article and original third-party datasets.

## Release sequence

1. Freeze the accepted-manuscript code as a tagged release.
2. Re-run `scripts/validate_package.py` and regenerate `MANIFEST.csv` and `SHA256SUMS.txt`.
3. Remove anonymous-review wording and add authors, affiliations, funder metadata and licence.
4. Deposit the release and reserve or mint the DOI.
5. Test the public landing page and download from a logged-out browser.
6. Add the DOI to the manuscript Code availability statement and software citation.

No repository record or DOI is asserted in this anonymous package.

