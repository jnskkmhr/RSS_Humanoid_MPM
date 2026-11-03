# aFluid RSS paper

`paper_template.tex` is the main document. It keeps the packages, title,
authors, section order, and bibliography setup. Edit the paper text in
`sections/`, following the same `\input{sections/...}` layout as the sibling
`final_report` repository.

```text
paper_template.tex
sections/
  abstract.tex             Abstract
  introduction.tex         Original introduction and planning notes
  introduction_draft.tex   Active introduction draft
  template_section.tex     Existing "Section" placeholder and its subsections
  rss_citations.tex        RSS citation and hyperlink examples
  conclusion.tex           Conclusion
  acknowledgments.tex      Acknowledgments
references.bib             Original bibliography entries
introduction_references.bib References for the introduction draft
IEEEtran.cls               Conference document class
paper_template.pdf         Compiled paper
```

Each section file includes its heading or environment. Subsections stay
with their parent section. The main document uses `\input` so the file
boundaries do not force page breaks. The trailing `%` on each input line
suppresses additional whitespace at the file boundary.

Compile `paper_template.tex` from the repository root with pdfLaTeX and
BibTeX, or select it as the main document in Overleaf. With `latexmk`
installed, build into a separate directory to preserve the checked-in PDF:

```sh
latexmk -pdf -outdir=/tmp/rss_26_fluid-build paper_template.tex
```

The main document currently includes `sections/introduction_draft.tex` and
loads both bibliography files. The original introduction remains available in
`sections/introduction.tex`. The checked-in PDF contains the current draft;
rebuild it after source edits. The remaining template sections are still included.
