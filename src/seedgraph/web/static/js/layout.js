// Layout toggle skeleton (docs/V3_UI_DESIGN.md §2.4): split | pdf | notes.
// Markup contract (the paper page, U8):
//   <div class="sg-layout" data-layout="split"> ... </div>
//   <button type="button" data-layout-set="pdf">PDF</button>
// Clicking a [data-layout-set] button sets data-layout on the nearest .sg-layout (or
// the first one on the page) and records ?layout= in the URL without a reload, so a
// reload or a shared link keeps the layout. The CSS in app.css does the rest.
(function () {
  "use strict";
  var LAYOUTS = ["split", "pdf", "notes"];

  function container(from) {
    return (from && from.closest(".sg-layout")) || document.querySelector(".sg-layout");
  }

  function apply(el, layout) {
    if (!el || LAYOUTS.indexOf(layout) === -1) return;
    el.setAttribute("data-layout", layout);
    document.querySelectorAll("[data-layout-set]").forEach(function (b) {
      b.setAttribute("aria-pressed", String(b.getAttribute("data-layout-set") === layout));
    });
    var url = new URL(window.location.href);
    url.searchParams.set("layout", layout);
    window.history.replaceState(null, "", url.toString());
  }

  document.addEventListener("click", function (evt) {
    var btn = evt.target.closest ? evt.target.closest("[data-layout-set]") : null;
    if (!btn) return;
    evt.preventDefault();
    apply(container(btn), btn.getAttribute("data-layout-set"));
  });

  document.addEventListener("DOMContentLoaded", function () {
    var el = container(null);
    var wanted = new URL(window.location.href).searchParams.get("layout");
    if (el && wanted) apply(el, wanted);
  });

  window.seedgraph = window.seedgraph || {};
  window.seedgraph.setLayout = function (layout) { apply(container(null), layout); };
})();
