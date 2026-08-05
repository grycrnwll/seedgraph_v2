// seedgraph web UI — shared front-end helpers (Stage A skeleton).
//
// Minimal, dependency-free, no build step. Stage B mutating forms read the CSRF
// token from the <meta name="csrf-token"> tag injected by base.html and post it in
// the hidden `_csrf` field; this exposes a tiny helper so screens stay DRY.

(function (global) {
  "use strict";

  function csrfToken() {
    const meta = document.querySelector('meta[name="csrf-token"]');
    return meta ? meta.getAttribute("content") || "" : "";
  }

  // Progressive enhancement: drag-and-drop over the native <input type=file>.
  // The form still works without JS (the input is a plain file chooser).
  function wireDropzones() {
    document.querySelectorAll("[data-dropzone]").forEach(function (zone) {
      var input = document.getElementById(zone.getAttribute("data-input"));
      var count = zone.querySelector("[data-dropzone-count]");
      if (!input) return;
      function show() {
        if (count) count.textContent = input.files.length
          ? input.files.length + " file(s) selected" : "";
      }
      ["dragenter", "dragover"].forEach(function (ev) {
        zone.addEventListener(ev, function (e) { e.preventDefault(); zone.classList.add("dropzone--over"); });
      });
      ["dragleave", "drop"].forEach(function (ev) {
        zone.addEventListener(ev, function (e) { e.preventDefault(); zone.classList.remove("dropzone--over"); });
      });
      zone.addEventListener("drop", function (e) {
        var dt = new DataTransfer();
        Array.prototype.forEach.call(e.dataTransfer.files, function (f) { dt.items.add(f); });
        input.files = dt.files;
        show();
      });
      input.addEventListener("change", show);
    });
  }
  if (document.readyState !== "loading") wireDropzones();
  else document.addEventListener("DOMContentLoaded", wireDropzones);

  global.Seedgraph = { csrfToken: csrfToken };
})(window);

// Corpus conversion-status poller (background marker queue). While any row is
// queued/converting, poll the gated status endpoint every ~2s and live-update the
// status cells; when the queue drains, reload once to surface the converted rows
// (markdown links etc.), then stop. Dependency-free; the page works without it.
(function () {
  "use strict";
  function pending() {
    return Array.prototype.filter.call(
      document.querySelectorAll("[data-conv-status]"),
      function (c) {
        var s = (c.getAttribute("data-state") || "").trim();
        return s === "queued" || s === "converting";
      }
    );
  }
  // Recompute the aggregate conversion bar client-side from the same per-row states
  // the poller just refreshed — mirrors progress.conversion_summary (converted =
  // rows with markdown; pending = queued + converting; total = converted + pending).
  function updateConvBar() {
    var bar = document.querySelector("[data-conv-bar]");
    var label = document.querySelector("[data-conv-counts]");
    if (!bar && !label) return;
    var c = { converted: 0, converting: 0, queued: 0, failed: 0 };
    document.querySelectorAll("[data-conv-status]").forEach(function (cell) {
      var s = (cell.getAttribute("data-state") || "").trim();
      if (s === "converted") c.converted++;
      else if (s === "converting") c.converting++;
      else if (s === "queued") c.queued++;
      else if (s === "failed") c.failed++;
    });
    var total = c.converted + c.queued + c.converting;
    if (bar) {
      bar.max = Math.max(c.converted, total);
      bar.value = c.converted;
    }
    if (label) {
      var txt = c.converted + "/" + total + " converted";
      if (c.converting) txt += " · " + c.converting + " converting";
      if (c.queued) txt += " · " + c.queued + " queued";
      if (c.failed) txt += " · " + c.failed + " failed";
      label.textContent = txt;
    }
  }
  function wireConvPoller() {
    var root = document.querySelector("[data-corpus-slug]");
    if (!root) return;
    var slug = root.getAttribute("data-corpus-slug");
    if (pending().length === 0) return; // nothing pending on this page
    var sawPending = true; // guard the one-shot reload against loops
    var timer = setInterval(function () {
      fetch("/ui/projects/" + encodeURIComponent(slug) + "/upload/status", {
        headers: { Accept: "application/json" }
      })
        .then(function (r) { return r.ok ? r.json() : {}; })
        .then(function (map) {
          document.querySelectorAll("[data-conv-status]").forEach(function (cell) {
            var tr = cell.closest("tr");
            var wid = tr ? tr.getAttribute("data-work-id") : null;
            if (!wid) return;
            var st = map[wid];
            if (st) {
              cell.setAttribute("data-state", st.state);
              cell.className = "conv conv--" + st.state;
              cell.textContent = st.state === "failed" && st.detail
                ? "failed: " + st.detail : st.state;
            } else if (["queued", "converting"].indexOf(
              cell.getAttribute("data-state")) !== -1) {
              // Was pending, now gone from the map => finished converting.
              cell.setAttribute("data-state", "converted");
              cell.className = "conv conv--converted";
              cell.textContent = "converted";
            }
          });
          updateConvBar();
          if (pending().length === 0) {
            clearInterval(timer);
            if (sawPending) window.location.reload();
          }
        })
        .catch(function () {});
    }, 2000);
  }
  if (document.readyState !== "loading") wireConvPoller();
  else document.addEventListener("DOMContentLoaded", wireConvPoller);
})();
