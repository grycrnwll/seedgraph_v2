// seedgraph web UI — run-events polling helper (Track 2).
//
// Polls GET /api/.../runs/{run_id}/events?after={seq} (~1.5s; no WebSockets),
// threading the monotonic `after` seq cursor, until the run reaches a terminal
// status (finished | failed). Dependency-free; loaded with a plain <script>.

(function (global) {
  "use strict";

  var TERMINAL = { finished: true, failed: true };

  // watchRun(url, { onEvents, onStatus, onTerminal, intervalMs }) -> stop()
  //
  // The events endpoint returns { run_id, events: [...], status }. `onEvents` is
  // called with each new batch (seq-ordered); `onStatus` with the run's current
  // coarse status; `onTerminal` once when a terminal status is first observed.
  function watchRun(url, opts) {
    opts = opts || {};
    var intervalMs = opts.intervalMs || 1500;
    var after = -1; // -1 => return all events on the first poll
    var stopped = false;
    var timer = null;

    function schedule() {
      if (!stopped) timer = setTimeout(tick, intervalMs);
    }

    function tick() {
      if (stopped) return;
      fetch(url + "?after=" + after, { credentials: "same-origin" })
        .then(function (resp) {
          if (!resp.ok) {
            schedule();
            return null;
          }
          return resp.json();
        })
        .then(function (data) {
          if (!data) return;
          var events = data.events || [];
          if (Array.isArray(events) && events.length) {
            after = events[events.length - 1].seq;
            if (opts.onEvents) opts.onEvents(events);
          }
          if (opts.onStatus) opts.onStatus(data.status);
          if (data.status && TERMINAL[data.status]) {
            stop();
            if (opts.onTerminal) opts.onTerminal(data.status);
            return;
          }
          schedule();
        })
        .catch(function () {
          schedule();
        });
    }

    function stop() {
      stopped = true;
      if (timer) clearTimeout(timer);
    }

    timer = setTimeout(tick, intervalMs);
    return stop;
  }

  global.SeedgraphPoll = { watchRun: watchRun };
})(window);
