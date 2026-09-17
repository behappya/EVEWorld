/* ============================================================
   EVEWorld project page — interactions (v2)

   Hero video reel, tabbed failure-mode widget, figure deck
   carousel, scroll progress, hover-to-play videos, qualitative
   filter, nav highlighting, reveal-on-scroll and BibTeX copy.
   Honors prefers-reduced-motion.

   Adapted from the LIBERO-Recover project page (same visual
   system); the level widget here switches still images rather
   than videos, and the flip-card explorer is not used.
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
  if (reel && !reduceMotion) {
    reel.querySelectorAll(".reel-row").forEach(function (row) {
      row.innerHTML += row.innerHTML; /* seamless -50% marquee loop */
    });
    var reelVideos = reel.querySelectorAll("video");
    /* the reel tiles carry preload="none", so borrow the qualitative posters to
       paint them immediately instead of showing black until playback starts */
    reelVideos.forEach(function (v) {
      var s = v.getAttribute("src");
      if (s) v.poster = s.replace("videos/gr1/", "videos/posters/").replace(/([^/]+)\/([^/]+)\.mp4$/, "$1-$2.jpg");
    });
    if ("IntersectionObserver" in window) {
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

  /* ---------- generic hover-to-play video boxes ---------- */

  document.querySelectorAll("[data-video]").forEach(function (box) {
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

  /* pause offscreen videos to keep the page light */
  if ("IntersectionObserver" in window) {
    var vio = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        var v = en.target.querySelector("video");
        if (!v) return;
        if (!en.isIntersecting && !v.paused) {
          v.pause();
          en.target.classList.remove("playing");
        }
      });
    }, { rootMargin: "120px" });
    document.querySelectorAll("[data-video]").forEach(function (b) { vio.observe(b); });
  }

  /* ---------- failure-mode widget ----------
     Keys are the CSS accent variables (--l4 / --l3 / --l1) so that
     the stage accent can be set from the key directly. */

  var MODES = {
    l4: {
      badge: 'Duplication &middot; N<sub>t</sub> &gt; N<sub>0</sub>',
      name: "Two targets where there should be one",
      blurb: 'Frame-level reconstruction admits a shortcut: leave a copy of the target at the destination, or keep the source instance in place while a second one appears at the goal. The endpoint cue is satisfied, so the loss stays low &mdash; but the scene now contains an instance that the demonstration never had.',
      caps: ["instance conservation", "spatial emphasis &mdash; IGR"],
      statV: 'N<sub>t</sub> &gt; N<sub>0</sub>',
      statL: "the target count exceeds the initial frame &mdash; the goal is reached by spawning an additional instance"
    },
    l3: {
      badge: 'Disappearance &middot; N<sub>t</sub> &lt; N<sub>0</sub>',
      name: "The target leaves before the task closes",
      blurb: 'The opposite shortcut. Rather than spawning a second instance, the rollout lets the target fade out once the endpoint cue is satisfied, or drops it in the middle of the interaction. The frames that matter still look acceptable, but the instance the instruction names is no longer present &mdash; and a count-based endpoint check never notices.',
      caps: ["instance conservation", "cross-frame tracking &mdash; TIA"],
      statV: 'N<sub>t</sub> &lt; N<sub>0</sub>',
      statL: "the target disappears mid-interaction &mdash; the instruction names an instance the rollout no longer contains"
    },
    l1: {
      badge: 'Distortion &middot; N<sub>t</sub> = N<sub>0</sub>',
      name: "The count is right, the instance is not",
      blurb: 'The count is preserved at every sampled timestamp, yet the target deforms, changes appearance, or jumps between frames. Nothing in a frame-level objective ties one frame&rsquo;s instance to the next, so identity drifts freely while every individual frame stays plausible. This is the failure mode TIA is designed to prevent.',
      caps: ["instance identity", "cross-frame alignment &mdash; TIA"],
      statV: 'N<sub>t</sub> = N<sub>0</sub>',
      statL: "the target count is conserved but the instance itself is not &mdash; each frame holds a target, not the same target"
    }
  };

  var lvlTabs = document.querySelectorAll(".lvl-tab");
  var lvlStage = document.querySelector(".lvl-stage");
  var lvlBadge = document.getElementById("lvlBadge");
  var lvlName = document.getElementById("lvlName");
  var lvlBlurb = document.getElementById("lvlBlurb");
  var lvlCap = document.getElementById("lvlCap");
  var lvlStatV = document.getElementById("lvlStatV");
  var lvlStatL = document.getElementById("lvlStatL");

  function setLevel(key) {
    var d = MODES[key];
    if (!d || !lvlStage) return;

    lvlTabs.forEach(function (t) {
      var active = t.getAttribute("data-level") === key;
      t.classList.toggle("is-active", active);
      t.setAttribute("aria-selected", active ? "true" : "false");
    });

    /* stage accent + all derived colors follow --lvl-c */
    lvlStage.style.setProperty("--lvl-c", "var(--" + key + ")");

    /* innerHTML, not textContent: the badges and stats carry <sub> subscripts */
    lvlBadge.innerHTML = d.badge;
    lvlName.textContent = d.name;
    lvlBlurb.innerHTML = d.blurb;
    lvlCap.innerHTML = d.caps.map(function (c) {
      return '<span class="cap-chip">' + c + "</span>";
    }).join("");
    lvlStatV.innerHTML = d.statV;
    if (lvlStatL) { lvlStatL.innerHTML = d.statL; }

    document.querySelectorAll(".lvl-stage-media.still img").forEach(function (im) {
      im.hidden = im.getAttribute("data-for") !== key;
    });
  }

  lvlTabs.forEach(function (t) {
    t.addEventListener("click", function () { setLevel(t.getAttribute("data-level")); });
  });
  if (lvlTabs.length) {
    setLevel(document.querySelector(".lvl-tab.is-active").getAttribute("data-level"));
  }

  /* ---------- figure deck ---------- */

  var deck = document.getElementById("figdeck");
  if (deck) {
    var panels = deck.querySelectorAll(".figdeck-panel");
    var rail = document.getElementById("deckRail");
    var prev = document.getElementById("deckPrev");
    var next = document.getElementById("deckNext");
    var counter = document.getElementById("deckCounter");
    var idx = 0;

    panels.forEach(function (_, i) {
      var dot = document.createElement("button");
      dot.className = "figdeck-dot" + (i === 0 ? " is-active" : "");
      dot.type = "button";
      dot.setAttribute("aria-label", "Show figure " + (i + 1));
      dot.addEventListener("click", function () { show(i); });
      rail.appendChild(dot);
    });
    var dots = rail.querySelectorAll(".figdeck-dot");

    function show(i) {
      idx = Math.max(0, Math.min(panels.length - 1, i));
      panels.forEach(function (p, j) { p.classList.toggle("is-active", j === idx); });
      dots.forEach(function (d, j) { d.classList.toggle("is-active", j === idx); });
      counter.textContent = (idx + 1) + " / " + panels.length;
      prev.disabled = idx === 0;
      next.disabled = idx === panels.length - 1;
    }
    prev.addEventListener("click", function () { show(idx - 1); });
    next.addEventListener("click", function () { show(idx + 1); });
    deck.addEventListener("keydown", function (e) {
      if (e.key === "ArrowLeft") show(idx - 1);
      if (e.key === "ArrowRight") show(idx + 1);
    });
    deck.setAttribute("tabindex", "0");
    show(0);
  }

  /* ---------- qualitative filter ---------- */

  var pills = document.querySelectorAll(".filter-pill");
  pills.forEach(function (pill) {
    pill.addEventListener("click", function () {
      pills.forEach(function (p) { p.classList.remove("active"); });
      pill.classList.add("active");
      var suite = pill.getAttribute("data-suite");
      document.querySelectorAll(".demo-card").forEach(function (card) {
        var showIt = suite === "all" || card.getAttribute("data-suite") === suite;
        card.classList.toggle("hidden", !showIt);
      });
    });
  });

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

  /* ---------- BibTeX copy ---------- */

  var btn = document.getElementById("copyBibtex");
  if (btn) {
    btn.addEventListener("click", function () {
      var text = document.getElementById("bibtex").textContent.trim();
      var done = function () {
        btn.textContent = "Copied ✓";
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
})();
