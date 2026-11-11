# PLOS ONE submission package

| File | Purpose |
|---|---|
| `manuscript.tex` / `manuscript.pdf` | Manuscript in PLOS layout: line numbers, numbered references, figure captions in the text, no embedded figures |
| `figures/Fig1.tif` … `Fig5.tif` | Figures as separate 600-dpi TIFF files (LZW), numbered in order of citation |
| `cover_letter.md` | Cover letter draft |
| `fee_assistance_request.md` | Draft text for the Publication Fee Assistance application |

Build: `latexmk -pdf manuscript.tex` (uses `../tables/` and `../refs.bib`).

Not verified against the journal's current instructions (the PLOS site could not be reached when this was
prepared): the official LaTeX template and `plos2015.bst` reference style (this version uses `unsrtnat`, a
numbered style in citation order), figure size limits, and whether funding, competing-interest and
data-availability statements must be entered only in the submission form. Check
https://journals.plos.org/plosone/s/submission-guidelines before uploading, and run the figures through the
journal's figure-checking tool if one is offered.
