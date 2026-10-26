// PDF pane controller for the paper page (docs/V3_UI_DESIGN.md §2.4, §3.3).
//
// - Renders the paper's PDF with the vendored PDF.js generic viewer (pages render lazily as
//   they scroll into view; the text layer is on). isEvalSupported is false (CSP).
// - A click on a quote or its "p. N" chip ([data-quote] with data-page, data-page-end,
//   data-how, data-needle) scrolls to the page, waits for its text layer, folds the
//   text-layer text with fold.js and highlights the exact passage as mark.sg-hl. If the
//   passage is not on the page(s), the neighbouring pages are tried once, then the page is
//   outlined (page-only highlight) with a notice. A quote without page information shows a
//   notice and never guesses a page.
// - Toolbar: previous / next page, page number, zoom, find (PDF.js find controller).
// - Remembers the layout toggle in localStorage (wrapped in try/catch; the query parameter
//   wins, and the page works without storage).
//
// Event handlers are delegated from document, because htmx replaces the notes pane.

import * as pdfjsLib from "/static/vendor/pdfjs/pdf.min.mjs";
import { fold, findPassage } from "/static/js/fold.js";

const VENDOR = "/static/vendor/pdfjs/";
const LAYOUT_KEY = "seedgraph.paper.layout";
const NOT_FOUND = "exact passage not found on this page; showing the page";
const NO_PAGE = "no page information for this quote: re-convert the paper to get pages";
const NOT_READY = "the PDF is still loading; try again in a moment";

globalThis.pdfjsLib = pdfjsLib; // pdf_viewer.mjs reads it, so it is imported dynamically below
pdfjsLib.GlobalWorkerOptions.workerSrc = "/static/vendor/pdfjs/pdf.worker.min.mjs";

const state = { viewer: null, eventBus: null, doc: null, last: null };
const waiters = new Map(); // pageNumber -> [resolve]

// --- layout memory ---------------------------------------------------------------------

function storageGet(key) {
  try {
    return window.localStorage.getItem(key);
  } catch (e) {
    return null;
  }
}

function storageSet(key, value) {
  try {
    window.localStorage.setItem(key, value);
  } catch (e) {
    /* storage unavailable: the query parameter still carries the layout */
  }
}

function rememberLayout() {
  const params = new URL(window.location.href).searchParams;
  const saved = storageGet(LAYOUT_KEY);
  if (!params.get("layout") && saved && window.seedgraph && window.seedgraph.setLayout) {
    window.seedgraph.setLayout(saved);
  }
  document.addEventListener("click", (evt) => {
    const btn = evt.target.closest ? evt.target.closest("[data-layout-set]") : null;
    if (btn) storageSet(LAYOUT_KEY, btn.getAttribute("data-layout-set"));
  });
}

// --- notices -----------------------------------------------------------------------------

function notice(text) {
  const el = document.getElementById("sg-pdf-notice");
  if (!el) return;
  el.textContent = text || "";
  el.hidden = !text;
}

// --- viewer ------------------------------------------------------------------------------

function pageView(n) {
  return state.viewer ? state.viewer.getPageView(n - 1) : null;
}

function textLayerOf(n) {
  const view = pageView(n);
  return view && view.div ? view.div.querySelector(".textLayer") : null;
}

function isRendered(n) {
  const layer = textLayerOf(n);
  return !!(layer && layer.childNodes.length && !layer.classList.contains("sg-stale"));
}

function waitForTextLayer(n, timeoutMs) {
  if (isRendered(n)) return Promise.resolve(true);
  return new Promise((resolve) => {
    const list = waiters.get(n) || [];
    list.push(resolve);
    waiters.set(n, list);
    setTimeout(() => resolve(isRendered(n)), timeoutMs || 8000);
  });
}

async function init() {
  rememberLayout();
  const container = document.getElementById("sg-pdf-container");
  if (!container) return;
  const viewerLib = await import(VENDOR + "pdf_viewer.mjs");
  const eventBus = new viewerLib.EventBus();
  const linkService = new viewerLib.PDFLinkService({ eventBus });
  const findController = new viewerLib.PDFFindController({ eventBus, linkService });
  const viewer = new viewerLib.PDFViewer({
    container,
    viewer: document.getElementById("sg-pdf-viewer"),
    eventBus,
    linkService,
    findController,
    textLayerMode: 1,
  });
  linkService.setViewer(viewer);
  state.viewer = viewer;
  state.eventBus = eventBus;

  eventBus.on("pagesinit", () => {
    viewer.currentScaleValue = "page-width";
    const count = document.querySelector("[data-pdf-page-count]");
    if (count) count.textContent = "of " + viewer.pagesCount;
    const input = document.querySelector("[data-pdf-page-input]");
    if (input) input.max = String(viewer.pagesCount);
  });
  eventBus.on("pagechanging", (evt) => {
    const input = document.querySelector("[data-pdf-page-input]");
    if (input) input.value = String(evt.pageNumber);
  });
  eventBus.on("textlayerrendered", (evt) => {
    const n = evt.pageNumber;
    const list = waiters.get(n) || [];
    waiters.delete(n);
    list.forEach((resolve) => resolve(true));
    // a re-render (zoom) drops the marks: put the last highlight back
    if (state.last && state.last.pages.indexOf(n) !== -1) applyHighlight(state.last, false);
  });

  try {
    const task = pdfjsLib.getDocument({
      url: container.getAttribute("data-pdf-url"),
      isEvalSupported: false,
      cMapUrl: VENDOR + "cmaps/",
      cMapPacked: true,
      standardFontDataUrl: VENDOR + "standard_fonts/",
    });
    state.doc = await task.promise;
    viewer.setDocument(state.doc);
    linkService.setDocument(state.doc, null);
  } catch (err) {
    notice("the PDF could not be opened: " + (err && err.message ? err.message : err));
  }
}

// --- highlight ---------------------------------------------------------------------------

function clearHighlights() {
  document.querySelectorAll("mark.sg-hl").forEach((mark) => {
    const parent = mark.parentNode;
    while (mark.firstChild) parent.insertBefore(mark.firstChild, mark);
    parent.removeChild(mark);
    parent.normalize();
  });
  document.querySelectorAll(".page.sg-hl-page").forEach((el) => el.classList.remove("sg-hl-page"));
}

// Concatenate the text nodes of the pages' text layers, remembering where each node starts.
function collect(pages) {
  const nodes = [];
  let raw = "";
  pages.forEach((n) => {
    const layer = textLayerOf(n);
    if (!layer) return;
    const walker = document.createTreeWalker(layer, NodeFilter.SHOW_TEXT);
    let node = walker.nextNode();
    while (node) {
      nodes.push({ node, start: raw.length });
      raw += node.data;
      node = walker.nextNode();
    }
    raw += "\n";
  });
  return { raw, nodes };
}

function find(pages, needle) {
  const { raw, nodes } = collect(pages);
  if (!nodes.length) return null;
  const folded = fold(raw);
  const hits = findPassage(folded.text, needle);
  if (!hits.length) return null;
  const [rs, re] = folded.rawSpan(hits[0][0], hits[0][1]);
  return { nodes, rs, re };
}

// Wrap [rs, re) of the concatenated text in mark.sg-hl, one mark per text node.
function wrap(found) {
  const marks = [];
  found.nodes.forEach(({ node, start }) => {
    const end = start + node.data.length;
    const s = Math.max(found.rs, start);
    const e = Math.min(found.re, end);
    if (s >= e) return;
    const range = document.createRange();
    range.setStart(node, s - start);
    range.setEnd(node, e - start);
    const mark = document.createElement("mark");
    mark.className = "sg-hl";
    range.surroundContents(mark);
    marks.push(mark);
  });
  return marks;
}

function pageOnly(page) {
  const view = pageView(page);
  if (view && view.div) {
    view.div.classList.add("sg-hl-page");
    view.div.scrollIntoView({ block: "start" });
  }
  notice(NOT_FOUND);
}

function applyHighlight(target, scroll) {
  clearHighlights();
  const found = find(target.pages, target.needle);
  if (!found) return false;
  const marks = wrap(found);
  if (scroll && marks.length) marks[0].scrollIntoView({ block: "center" });
  return marks.length > 0;
}

async function showQuote(el) {
  const page = parseInt(el.getAttribute("data-page") || "", 10);
  const pageEnd = parseInt(el.getAttribute("data-page-end") || "", 10) || page;
  const needle = el.getAttribute("data-needle") || "";
  if (!page || el.getAttribute("data-how") === "none") {
    notice(NO_PAGE);
    return;
  }
  if (!state.viewer || !state.viewer.pagesCount) {
    notice(NOT_READY);
    return;
  }
  if (window.seedgraph && window.seedgraph.setLayout) {
    const layout = document.querySelector(".sg-layout");
    if (layout && layout.getAttribute("data-layout") === "notes") window.seedgraph.setLayout("split");
  }
  notice("");
  const pages = [];
  for (let n = page; n <= pageEnd && n <= state.viewer.pagesCount; n++) pages.push(n);
  state.viewer.scrollPageIntoView({ pageNumber: page });
  await Promise.all(pages.map((n) => waitForTextLayer(n)));
  state.last = { pages, needle };
  if (needle && applyHighlight(state.last, true)) return;
  // neighbouring pages, once
  const around = [page - 1, ...pages, pageEnd + 1].filter((n) => n >= 1 && n <= state.viewer.pagesCount);
  for (const n of around) {
    if (pages.indexOf(n) === -1) {
      state.viewer.scrollPageIntoView({ pageNumber: n });
      await waitForTextLayer(n, 4000);
    }
  }
  state.last = { pages: around, needle };
  if (needle && applyHighlight(state.last, true)) return;
  state.last = null;
  pageOnly(page);
}

// --- toolbar -----------------------------------------------------------------------------

function toolbar(action) {
  const v = state.viewer;
  if (!v || !v.pagesCount) return;
  if (action === "prev") v.previousPage();
  else if (action === "next") v.nextPage();
  else if (action === "zoom-in") v.increaseScale();
  else if (action === "zoom-out") v.decreaseScale();
}

document.addEventListener("click", (evt) => {
  if (!evt.target.closest) return;
  const quote = evt.target.closest("[data-quote]");
  if (quote) {
    evt.preventDefault();
    showQuote(quote);
    return;
  }
  const btn = evt.target.closest("[data-pdf-action]");
  if (btn) {
    evt.preventDefault();
    toolbar(btn.getAttribute("data-pdf-action"));
  }
});

document.addEventListener("change", (evt) => {
  const input = evt.target.closest ? evt.target.closest("[data-pdf-page-input]") : null;
  if (input && state.viewer && state.viewer.pagesCount) {
    const n = Math.min(Math.max(parseInt(input.value, 10) || 1, 1), state.viewer.pagesCount);
    state.viewer.currentPageNumber = n;
  }
});

document.addEventListener("keydown", (evt) => {
  const input = evt.target.closest ? evt.target.closest("[data-pdf-find]") : null;
  if (!input || evt.key !== "Enter" || !state.eventBus) return;
  evt.preventDefault();
  state.eventBus.dispatch("find", {
    source: input,
    type: "again",
    query: input.value,
    caseSensitive: false,
    entireWord: false,
    highlightAll: true,
    findPrevious: evt.shiftKey,
    matchDiacritics: false,
  });
});

window.seedgraph = window.seedgraph || {};
window.seedgraph.pdfPane = { showQuote, state };

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
