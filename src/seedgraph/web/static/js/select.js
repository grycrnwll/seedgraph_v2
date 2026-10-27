// Selection capture for quote-backed claims (docs/V3_UI_DESIGN.md §5).
//
// PDF pane: a selection inside a PDF.js text layer shows "Add claim from selection"; the
// click posts {page, text} to /papers/{id}/anchor (library.pages.anchor_selection). The
// page is the one holding the selection's start node.
//
// Text view: each chunk carries data-offset, the code-point offset of its text in the
// markdown. A selection there posts {start, end} (code points: JavaScript offsets are
// UTF-16, so they are converted with Array.from) to /papers/{id}/anchor-text.
//
// Either answer (the add form with the exact markdown slice, or a refusal message) is
// swapped into #sg-add-claim. The CSRF header comes from csrf.js (htmx:configRequest).

(function () {
  "use strict";

  let pending = null; // what the visible button will send

  function hideBars() {
    document.querySelectorAll("#sg-selection-bar, [data-text-selection-bar]").forEach((bar) => {
      bar.hidden = true;
    });
  }

  function elementOf(node) {
    return node && node.nodeType === Node.TEXT_NODE ? node.parentElement : node;
  }

  function codePoints(text, utf16Offset) {
    return Array.from(text.slice(0, utf16Offset)).length;
  }

  // The markdown offset of a DOM position inside a text-view chunk, or null.
  function textViewOffset(node, offset) {
    const chunk = elementOf(node) && elementOf(node).closest("[data-offset]");
    if (!chunk) return null;
    let before = 0;
    const walker = document.createTreeWalker(chunk, NodeFilter.SHOW_TEXT);
    let t = walker.nextNode();
    while (t) {
      if (t === node) return parseInt(chunk.getAttribute("data-offset"), 10) + before + codePoints(t.data, offset);
      before += Array.from(t.data).length;
      t = walker.nextNode();
    }
    return parseInt(chunk.getAttribute("data-offset"), 10) + (offset ? Array.from(chunk.textContent).length : 0);
  }

  function capture() {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || !sel.rangeCount) {
      hideBars();
      pending = null;
      return;
    }
    const range = sel.getRangeAt(0);
    const startEl = elementOf(range.startContainer);
    if (!startEl) return;

    const page = startEl.closest(".textLayer") && startEl.closest(".page[data-page-number]");
    if (page) {
      const text = sel.toString();
      if (!text.trim()) return;
      const bar = document.getElementById("sg-selection-bar");
      const button = bar && bar.querySelector("[data-add-from-selection]");
      if (!button) return;
      pending = { url: button.getAttribute("data-anchor-url"),
                  values: { page: page.getAttribute("data-page-number"), text } };
      bar.hidden = false;
      return;
    }

    const view = startEl.closest("[data-text-view]");
    if (view) {
      const start = textViewOffset(range.startContainer, range.startOffset);
      const end = textViewOffset(range.endContainer, range.endOffset);
      if (start === null || end === null || end <= start) return;
      const bar = view.querySelector("[data-text-selection-bar]");
      if (!bar) return;
      pending = { url: view.getAttribute("data-anchor-url"), values: { start, end } };
      bar.hidden = false;
    }
  }

  function send() {
    if (!pending || !window.htmx) return;
    window.htmx.ajax("POST", pending.url, { target: "#sg-add-claim", swap: "innerHTML", values: pending.values })
      .then(() => {
        const box = document.getElementById("sg-add-claim");
        if (box) box.scrollIntoView({ block: "nearest" });
      });
    hideBars();
  }

  document.addEventListener("mouseup", () => setTimeout(capture, 0));
  document.addEventListener("keyup", (evt) => {
    if (evt.shiftKey || evt.key === "Shift") setTimeout(capture, 0);
  });

  document.addEventListener("click", (evt) => {
    if (!evt.target.closest) return;
    if (evt.target.closest("[data-add-from-selection], [data-add-from-text]")) {
      evt.preventDefault();
      send();
      return;
    }
    if (evt.target.closest("[data-anchor-cancel]")) {
      evt.preventDefault();
      const box = document.getElementById("sg-add-claim");
      if (box) box.replaceChildren();
    }
  });

  // after a quote-backed claim is saved, the add form has done its job
  document.addEventListener("htmx:afterSwap", (evt) => {
    if (evt.detail && evt.detail.target && evt.detail.target.id === "sg-notes") {
      const box = document.getElementById("sg-add-claim");
      const form = box && box.querySelector("[data-anchor-form]");
      if (form && evt.detail.requestConfig && evt.detail.requestConfig.elt === form) box.replaceChildren();
    }
  });
})();

// Auto-grow for the notes pane's long fields (textarea[data-autogrow]): where CSS
// `field-sizing: content` is not supported, the height follows the text. Delegated on the
// document so it survives htmx swaps of #sg-notes; also sized when a <details> opens (a
// closed one has no layout) and after each htmx settle. The height is set through the
// CSSOM (el.style), which the CSP allows; no style attribute is written as markup.
(function () {
  "use strict";

  const native = typeof CSS !== "undefined" && CSS.supports && CSS.supports("field-sizing", "content");

  function grow(el) {
    if (native || !el || !el.matches || !el.matches("textarea[data-autogrow]")) return;
    if (!el.offsetParent) return; // hidden (closed details): sized when shown
    el.style.height = "auto";
    el.style.height = el.scrollHeight + 2 + "px";
  }

  function growWithin(root) {
    if (native || !root || !root.querySelectorAll) return;
    root.querySelectorAll("textarea[data-autogrow]").forEach(grow);
  }

  document.addEventListener("input", (evt) => grow(evt.target));
  document.addEventListener("toggle", (evt) => growWithin(evt.target), true);
  document.addEventListener("htmx:afterSettle", (evt) => growWithin(evt.target));
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => growWithin(document));
  } else {
    growWithin(document);
  }
})();
