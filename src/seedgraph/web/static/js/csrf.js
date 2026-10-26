// CSRF header hook (docs/V3_UI_DESIGN.md §8). The per-process token is read from
// <meta name="csrf-token"> (no inline script) and sent as X-CSRF-Token on every
// htmx request. A 403 that carries the restart hint shows the reload banner.
(function () {
  "use strict";
  var RESTART_HINT = "seedgraph restarted; reload the page";

  function token() {
    var meta = document.querySelector('meta[name="csrf-token"]');
    return meta ? meta.getAttribute("content") : "";
  }

  function showBanner(text) {
    var banner = document.getElementById("sg-banner");
    if (!banner) return;
    banner.textContent = text;
    banner.hidden = false;
  }

  document.addEventListener("htmx:configRequest", function (evt) {
    evt.detail.headers["X-CSRF-Token"] = token();
  });

  document.addEventListener("htmx:responseError", function (evt) {
    var xhr = evt.detail.xhr;
    if (xhr && xhr.status === 403 && String(xhr.responseText).indexOf(RESTART_HINT) !== -1) {
      showBanner(RESTART_HINT);
    }
  });

  window.seedgraph = window.seedgraph || {};
  window.seedgraph.csrfToken = token;
  window.seedgraph.showBanner = showBanner;
})();
