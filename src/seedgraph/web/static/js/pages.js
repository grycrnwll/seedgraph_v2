// Small behaviours for the Library, Batch and Review pages (no inline script; CSP §8).
//   <button type="button" data-copy="ID">Copy</button>   copies the text of #ID
//   <div data-dropzone="INPUT_ID">                       dropping PDFs fills <input type=file id=INPUT_ID>
//   <input type="checkbox" data-select-all="NAME">       checks every checkbox named NAME
(function () {
  "use strict";

  function copyText(text, button) {
    function done(ok) {
      if (!button) return;
      var old = button.textContent;
      button.textContent = ok ? "Copied" : "Select and copy";
      window.setTimeout(function () { button.textContent = old; }, 1500);
    }
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(function () { done(true); }, function () { done(false); });
        return;
      }
    } catch (e) { /* fall through */ }
    done(false);
  }

  document.addEventListener("click", function (evt) {
    var btn = evt.target.closest ? evt.target.closest("[data-copy]") : null;
    if (!btn) return;
    var source = document.getElementById(btn.getAttribute("data-copy"));
    if (source) copyText(source.textContent.trim(), btn);
  });

  document.addEventListener("change", function (evt) {
    var box = evt.target;
    if (!box.matches || !box.matches("[data-select-all]")) return;
    var name = box.getAttribute("data-select-all");
    document.querySelectorAll('input[type="checkbox"][name="' + name + '"]').forEach(function (b) {
      b.checked = box.checked;
    });
  });

  function setupDropzone(zone) {
    var input = document.getElementById(zone.getAttribute("data-dropzone"));
    if (!input) return;
    ["dragenter", "dragover"].forEach(function (name) {
      zone.addEventListener(name, function (evt) {
        evt.preventDefault();
        zone.classList.add("sg-dropzone-over");
      });
    });
    ["dragleave", "drop"].forEach(function (name) {
      zone.addEventListener(name, function () { zone.classList.remove("sg-dropzone-over"); });
    });
    zone.addEventListener("drop", function (evt) {
      evt.preventDefault();
      if (!evt.dataTransfer || !evt.dataTransfer.files.length) return;
      try {
        input.files = evt.dataTransfer.files;
        input.dispatchEvent(new Event("change", { bubbles: true }));
      } catch (e) { /* older browsers: use the file picker */ }
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("[data-dropzone]").forEach(setupDropzone);
  });
})();
