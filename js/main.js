/* ============================================================
   EVEWorld project page — interactions (v3)

   Hero video reel, scroll progress, click-to-play video tiles,
   synchronized playback for the three featured comparison cards,
   the figure lightbox, nav highlighting, reveal-on-scroll,
   disclosure state sync and BibTeX copy.
   Honors prefers-reduced-motion.

   Adapted from the LIBERO-Recover project page (same visual
   system).
   ============================================================ */
(function () {
  "use strict";

  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------- scroll progress ---------- */

  var bar = document.getElementById("scrollBar");
  function updateProgress() {
    if (!bar) return;
    var h = document.documentElement;
    var max = h.scrollHeight - h.clientHeight;
    bar.style.width = (max > 0 ? (h.scrollTop / max) * 100 : 0) + "%";
  }

  /* ---------- hero reel: duplicate each row, then play ---------- */

  var reel = document.getElementById("heroReel");
  if (reel) {
    if (!reduceMotion) {
      reel.querySelectorAll(".reel-row").forEach(function (row) {
        row.innerHTML += row.innerHTML; /* seamless -50% marquee loop */
      });
    }
    var reelVideos = reel.querySelectorAll("video");
    /* the reel tiles carry preload="none", so paint a poster frame immediately
       instead of showing black until playback starts; every clip sits next to
       its own poster. Under reduced motion the posters are what the wall shows,
       since the marquee and autoplay stay off */
    reelVideos.forEach(function (v) {
      var s = v.getAttribute("src");
      if (s) v.poster = s.replace(/\.mp4$/, ".jpg");
    });
    if (!reduceMotion && "IntersectionObserver" in window) {
      var rio = new IntersectionObserver(function (entries) {
        entries.forEach(function (en) {
          reelVideos.forEach(function (v) {
            if (en.isIntersecting) {
              v.play().catch(function () { /* autoplay blocked — dark tiles */ });
            } else {
              v.pause();
            }
          });
        });
      }, { rootMargin: "60px" });
      rio.observe(reel);
    }
  }

  /* ---------- disclosure state exposed to assistive tech ---------- */

  document.querySelectorAll("details.acc, details.acc-item").forEach(function (d) {
    var summary = d.querySelector("summary");
    if (!summary) return;
    var sync = function () { summary.setAttribute("aria-expanded", d.open ? "true" : "false"); };
    d.addEventListener("toggle", sync);
    sync();
  });

  /* ---------- placeholders stay inert until the authors fill them in ---------- */

  document.querySelectorAll('a[aria-disabled="true"]').forEach(function (a) {
    a.addEventListener("click", function (e) { e.preventDefault(); });
  });

  /* ---------- generic click-to-play video tiles ---------- */

  document.querySelectorAll("[data-video]").forEach(function (box) {
    /* the featured comparison cards are driven as one group, below */
    if (box.closest(".demo-card.featured")) return;

    var video = box.querySelector("video");
    if (!video) return;

    var pinned = false;
    function start() {
      video.play().then(function () {
        box.classList.add("playing");
      }).catch(function () { /* autoplay blocked — poster stays */ });
    }
    function stop() {
      if (pinned) return;
      video.pause();
      box.classList.remove("playing");
    }
    function toggle() {
      pinned = !pinned;
      if (pinned) { start(); }
      else { video.pause(); box.classList.remove("playing"); }
    }

    if (!reduceMotion) {
      box.addEventListener("mouseenter", start);
      box.addEventListener("mouseleave", stop);
    }
    box.addEventListener("click", toggle);
    box.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); }
    });
    box.setAttribute("tabindex", "0");
    box.setAttribute("role", "button");
    box.setAttribute("aria-label", "Play demonstration video");
  });

  /* ---------- featured cards: the matched arms play together ---------- */

  var groups = [];

  function pauseGroup(st) {
    st.pinned = false;
    st.group.classList.remove("playing");
    st.tiles.forEach(function (t) { t.classList.remove("playing"); });
    st.videos.forEach(function (v) { v.pause(); });
  }

  function playGroup(st) {
    groups.forEach(function (other) { if (other !== st) pauseGroup(other); });
    st.pinned = true;
    st.group.classList.add("playing");
    st.tiles.forEach(function (t) { t.classList.add("playing"); });
    st.videos.forEach(function (v) { v.play().catch(function () { /* autoplay blocked */ }); });
  }

  document.querySelectorAll(".demo-card.featured .demo-tiles").forEach(function (group) {
    var st = {
      group: group,
      videos: Array.prototype.slice.call(group.querySelectorAll("video")),
      tiles: Array.prototype.slice.call(group.querySelectorAll(".demo-tile")),
      pinned: false
    };
    if (!st.videos.length) return;

    group.setAttribute("tabindex", "0");
    group.setAttribute("role", "button");
    group.setAttribute("aria-label", "Play all matched rollouts of this comparison");
    group.addEventListener("click", function () {
      if (st.pinned) { pauseGroup(st); } else { playGroup(st); }
    });
    group.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        if (st.pinned) { pauseGroup(st); } else { playGroup(st); }
      }
    });
    groups.push(st);
  });

  function groupFor(el) {
    for (var i = 0; i < groups.length; i++) {
      if (groups[i].group.contains(el)) return groups[i];
    }
    return null;
  }

  /* pause offscreen videos to keep the page light */
  if ("IntersectionObserver" in window) {
    var vio = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) return;
        var v = en.target.querySelector("video");
        if (v && !v.paused) v.pause();
        en.target.classList.remove("playing");
        var g = groupFor(en.target);
        if (g && g.pinned) pauseGroup(g);
      });
    }, { rootMargin: "120px" });
    document.querySelectorAll("[data-video]").forEach(function (b) { vio.observe(b); });
  }

  /* ---------- navbar active link + progress on scroll ---------- */

  var navLinks = document.querySelectorAll(".header-nav a");
  var sections = [];
  navLinks.forEach(function (a) {
    var id = a.getAttribute("href").slice(1);
    var el = document.getElementById(id);
    if (el) sections.push({ id: id, el: el, link: a });
  });

  function onScroll() {
    updateProgress();
    var pos = window.scrollY + 130;
    var current = sections[0] && sections[0].id;
    sections.forEach(function (s) { if (s.el.offsetTop <= pos) current = s.id; });
    navLinks.forEach(function (a) {
      a.classList.toggle("active", a.getAttribute("href") === "#" + current);
    });
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  /* ---------- reveal on scroll ---------- */

  if ("IntersectionObserver" in window && !reduceMotion) {
    var wio = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) { en.target.classList.add("in"); wio.unobserve(en.target); }
      });
    }, { rootMargin: "0px 0px -40px 0px" });
    document.querySelectorAll(".reveal").forEach(function (el) { wio.observe(el); });
  } else {
    document.querySelectorAll(".reveal").forEach(function (el) { el.classList.add("in"); });
  }

  /* ---------- plate lightbox (section 05) ---------- */

  var zoomLinks = document.querySelectorAll(".plate-zoom");
  if (zoomLinks.length) {
    var lb = document.createElement("div");
    lb.className = "lightbox";
    lb.setAttribute("role", "dialog");
    lb.setAttribute("aria-modal", "true");
    lb.setAttribute("aria-label", "Figure preview");
    var lbImg = document.createElement("img");
    lbImg.alt = "";
    var lbClose = document.createElement("button");
    lbClose.className = "lightbox-close";
    lbClose.type = "button";
    lbClose.setAttribute("aria-label", "Close the preview");
    lbClose.textContent = "\u00D7";
    lb.appendChild(lbImg);
    lb.appendChild(lbClose);
    document.body.appendChild(lb);

    var closeLb = function () {
      lb.classList.remove("open");
      document.body.classList.remove("lb-open");
      lbImg.removeAttribute("src");
    };
    var openLb = function (href, alt) {
      lbImg.src = href;
      lbImg.alt = alt || "";
      lb.classList.add("open");
      document.body.classList.add("lb-open");
      lbClose.focus();
    };

    zoomLinks.forEach(function (a) {
      a.addEventListener("click", function (e) {
        e.preventDefault();
        var img = a.querySelector("img");
        openLb(a.getAttribute("href"), img ? img.getAttribute("alt") : "");
      });
    });
    lbClose.addEventListener("click", closeLb);
    lb.addEventListener("click", function (e) { if (e.target === lb) closeLb(); });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && lb.classList.contains("open")) closeLb();
    });
  }

  /* ---------- BibTeX copy ---------- */

  var btn = document.getElementById("copyBibtex");
  if (btn) {
    btn.addEventListener("click", function () {
      var text = document.getElementById("bibtex").textContent.trim();
      var done = function () {
        btn.textContent = "Copied \u2713";
        btn.classList.add("copied");
        setTimeout(function () {
          btn.textContent = "Copy BibTeX";
          btn.classList.remove("copied");
        }, 1800);
      };
      function fallback() {
        var ta = document.createElement("textarea");
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand("copy"); done(); } catch (e) { /* noop */ }
        document.body.removeChild(ta);
      }
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done).catch(fallback);
      } else { fallback(); }
    });
  }

  /* ---------- qualitative comparison: viewport-gated playback ---------- */

  var cmpVideos = Array.prototype.slice.call(document.querySelectorAll("video[data-cmp]"));
  if (cmpVideos.length && "IntersectionObserver" in window && !reduceMotion) {
    cmpVideos.forEach(function (v) {
      /* once the viewer takes manual control, stop auto-playing this clip */
      v.addEventListener("pointerdown", function () { v.dataset.hold = "1"; });
    });
    var cmpIo = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        var v = en.target;
        if (en.isIntersecting) {
          if (!v.dataset.hold && v.paused) {
            var pr = v.play();
            if (pr && pr.catch) pr.catch(function () {});
          }
        } else if (!v.paused) {
          v.pause();
        }
      });
    }, { rootMargin: "120px 0px", threshold: 0.35 });
    cmpVideos.forEach(function (v) { cmpIo.observe(v); });
  }

  /* ---------- section 07 experiments: finding explorer ---------- */

  var fx = document.querySelector("[data-fx]");
  if (fx) {
    var fxTabs = Array.prototype.slice.call(fx.querySelectorAll("[data-fx-tab]"));
    var fxPanels = Array.prototype.slice.call(fx.querySelectorAll("[data-fx-panel]"));
    var fxNow = fx.querySelector("[data-fx-now]");
    var fxIndex = 0;

    function fxShow(i) {
      if (!fxPanels.length) return;
      fxIndex = (i % fxPanels.length + fxPanels.length) % fxPanels.length;
      fxPanels.forEach(function (p, n) {
        var on = n === fxIndex;
        if (on) { p.removeAttribute("hidden"); } else { p.setAttribute("hidden", ""); }
        p.classList.toggle("is-active", on);
      });
      fxTabs.forEach(function (t, n) {
        var on = n === fxIndex;
        t.classList.toggle("is-active", on);
        t.setAttribute("aria-selected", on ? "true" : "false");
        t.setAttribute("tabindex", on ? "0" : "-1");
      });
      if (fxNow) fxNow.textContent = String(fxIndex + 1);
    }

    fxTabs.forEach(function (t, n) {
      t.addEventListener("click", function () { fxShow(n); });
      t.addEventListener("keydown", function (e) {
        if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
        e.preventDefault();
        e.stopPropagation();
        var next = (n + (e.key === "ArrowRight" ? 1 : -1) + fxTabs.length) % fxTabs.length;
        fxShow(next);
        fxTabs[next].focus();
      });
    });

    var fxPrev = fx.querySelector("[data-fx-prev]");
    var fxNext = fx.querySelector("[data-fx-next]");
    if (fxPrev) fxPrev.addEventListener("click", function () { fxShow(fxIndex - 1); });
    if (fxNext) fxNext.addEventListener("click", function () { fxShow(fxIndex + 1); });

    /* arrow keys step through findings while the explorer is on screen, and
       only when the viewer is not typing or scrubbing a clip */
    var fxInView = false;
    if ("IntersectionObserver" in window) {
      var fxIo = new IntersectionObserver(function (entries) {
        entries.forEach(function (en) { fxInView = en.isIntersecting; });
      }, { threshold: 0.25 });
      fxIo.observe(fx);
    }
    document.addEventListener("keydown", function (e) {
      if (!fxInView) return;
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      var t = e.target;
      if (t && t.closest && t.closest("input, textarea, select, video, [contenteditable]")) return;
      e.preventDefault();
      fxShow(fxIndex + (e.key === "ArrowRight" ? 1 : -1));
    });
  }
})();
