/* Figure 1, interactive. Vanilla JS, no dependencies.
 *
 * Each .stage holds the panel SVG twice: the colour copy (.stage-img) and a
 * greyed copy (.stage-grey, CSS filter) stacked on top. The greyed copy is
 * shown only over the regions that are NOT active, through clip-path.
 *
 * Coordinates live in the HTML, in pt of the SVG viewBox:
 *   .stage[data-vb="W H"]           the viewBox size
 *   .zone[data-pt="x0 y0 x1 y1"]    hover/focus/tap target
 *   .zone[data-grey="x0 y0 x1 y1"]  region greyed when ANOTHER zone is active
 *                                   (defaults to data-pt)
 * This script converts them to percentages, so the zones follow any size.
 *
 * Active zone = mouse hover, else keyboard focus, else the zone locked by a
 * click or tap (a second click or tap, Escape, or a click outside unlocks).
 * With no active zone the panel is in full colour and its box shows the
 * state named by .stage[data-default].
 */
(function () {
  'use strict';

  function nums(str) {
    return String(str || '').trim().split(/\s+/).map(Number);
  }

  function pct(v) {
    return (Math.round(v * 1000) / 1000) + '%';
  }

  // rect in pt -> {l, t, r, b} in % of the stage
  function toPct(rect, W, H) {
    return { l: 100 * rect[0] / W, t: 100 * rect[1] / H, r: 100 * rect[2] / W, b: 100 * rect[3] / H };
  }

  // One rectangle -> inset(); several -> a single polygon that walks the
  // rectangles down their left edge (the joins have zero width, so the gaps
  // between the rectangles stay unclipped).
  function clipFor(rects) {
    if (rects.length === 1) {
      var q = rects[0];
      return 'inset(' + pct(q.t) + ' ' + pct(100 - q.r) + ' ' + pct(100 - q.b) + ' ' + pct(q.l) + ')';
    }
    var sorted = rects.slice().sort(function (a, b) { return a.t - b.t; });
    var pts = [];
    sorted.forEach(function (q) {
      pts.push(pct(q.l) + ' ' + pct(q.t), pct(q.r) + ' ' + pct(q.t),
               pct(q.r) + ' ' + pct(q.b), pct(q.l) + ' ' + pct(q.b));
    });
    return 'polygon(' + pts.join(', ') + ')';
  }

  function isKeyboardFocus(el) {
    try { return el.matches(':focus-visible'); } catch (e) { return true; }
  }

  function initStage(stage) {
    var vb = nums(stage.getAttribute('data-vb'));
    var W = vb[0], H = vb[1];
    var grey = stage.querySelector('.stage-grey');
    var zones = Array.prototype.slice.call(stage.querySelectorAll('.zone'));
    if (!zones.length || !grey || !(W > 0 && H > 0)) { return; }
    var box = document.getElementById(zones[0].getAttribute('aria-controls'));
    var states = box ? Array.prototype.slice.call(box.querySelectorAll('.state')) : [];
    var defaultKey = stage.getAttribute('data-default');

    var hover = null, kbd = null, locked = null;

    // position the zones and precompute each zone's greyed region
    zones.forEach(function (z) {
      var p = toPct(nums(z.getAttribute('data-pt')), W, H);
      z.style.left = pct(p.l);
      z.style.top = pct(p.t);
      z.style.width = pct(p.r - p.l);
      z.style.height = pct(p.b - p.t);
      z._grey = toPct(nums(z.getAttribute('data-grey') || z.getAttribute('data-pt')), W, H);
    });

    function render() {
      var active = hover || kbd || locked;
      if (active) {
        var others = zones.filter(function (z) { return z !== active; })
                          .map(function (z) { return z._grey; });
        var clip = clipFor(others);
        grey.style.clipPath = clip;
        grey.style.webkitClipPath = clip;
        stage.classList.add('is-greyed');
      } else {
        stage.classList.remove('is-greyed');
      }
      zones.forEach(function (z) {
        z.setAttribute('aria-pressed', z === locked ? 'true' : 'false');
      });
      var key = active ? active.getAttribute('data-key') : defaultKey;
      var tone = null;
      states.forEach(function (s) {
        var on = s.getAttribute('data-state') === key;
        s.classList.toggle('is-current', on);
        s.setAttribute('aria-hidden', on ? 'false' : 'true');
        if (on) { tone = s.getAttribute('data-tone'); }
      });
      if (box) {
        if (tone) { box.setAttribute('data-tone', tone); } else { box.removeAttribute('data-tone'); }
      }
    }

    zones.forEach(function (z) {
      z.addEventListener('pointerenter', function (ev) {
        if (ev.pointerType === 'touch') { return; }   // touch uses tap-to-toggle
        hover = z; render();
      });
      z.addEventListener('pointerleave', function (ev) {
        if (ev.pointerType === 'touch') { return; }
        if (hover === z) { hover = null; render(); }
      });
      z.addEventListener('focus', function () {
        if (isKeyboardFocus(z)) { kbd = z; render(); }
      });
      z.addEventListener('blur', function () {
        if (kbd === z) { kbd = null; render(); }
      });
      z.addEventListener('click', function () {
        locked = (locked === z) ? null : z;
        render();
      });
      z.addEventListener('keydown', function (ev) {
        if (ev.key === 'Escape' || ev.key === 'Esc') {
          locked = null; kbd = null; render();
        }
      });
    });

    // a click or tap anywhere outside the stage unlocks it
    document.addEventListener('click', function (ev) {
      if (locked && !stage.contains(ev.target)) { locked = null; render(); }
    });

    render();
  }

  function init() {
    Array.prototype.slice.call(document.querySelectorAll('.stage[data-vb]')).forEach(initStage);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else { init(); }
})();
