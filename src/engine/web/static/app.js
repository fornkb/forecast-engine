// Small progressive enhancements. Every page works without this file.
(function () {
  "use strict";

  // Planner name for the audit trail, kept in a cookie the server reads.
  const who = document.getElementById("planner");
  if (who) {
    who.addEventListener("change", () => {
      const value = who.value.trim().slice(0, 60);
      document.cookie = "planner=" + encodeURIComponent(value) + "; path=/; max-age=31536000; SameSite=Strict";
    });
  }

  // Disable buttons after a submit so one click cannot start two paid runs.
  document.querySelectorAll("form[data-busy]").forEach((form) => {
    form.addEventListener("submit", () => {
      // Deferred, so the pressed button's name and value are still sent.
      setTimeout(() => form.querySelectorAll("button").forEach((b) => (b.disabled = true)), 0);
    });
  });

  // Table rows that open a page when clicked anywhere (links inside still work).
  document.querySelectorAll("tr[data-href]").forEach((row) => {
    row.addEventListener("click", (event) => {
      if (event.target.closest("a, button, input, select")) return;
      window.location.href = row.getAttribute("data-href");
    });
  });

  // Job page: poll until the job finishes, then reload (the server redirects).
  const jobBox = document.querySelector("[data-job]");
  if (jobBox) {
    const id = jobBox.getAttribute("data-job");
    const seconds = jobBox.querySelector("[data-seconds]");
    const tick = async () => {
      try {
        const response = await fetch("/api/jobs/" + encodeURIComponent(id), { cache: "no-store" });
        const job = await response.json();
        if (job.status !== "running") {
          window.location.reload();
          return;
        }
        if (seconds) seconds.textContent = String(job.seconds);
      } catch (error) {
        /* server busy or restarting: keep trying */
      }
      setTimeout(tick, 1200);
    };
    setTimeout(tick, 800);
  }

  // Chart hover: a crosshair snaps to the nearest month, one tooltip lists every value.
  const fmt = (v) => (v === null || v === undefined ? "–" : Math.round(v).toLocaleString());
  document.querySelectorAll("figure[data-chart]").forEach((figure) => {
    let data;
    try {
      data = JSON.parse(figure.getAttribute("data-chart"));
    } catch (error) {
      return;
    }
    const svg = figure.querySelector("svg");
    const hit = svg && svg.querySelector(".hit");
    const cross = svg && svg.querySelector(".cross");
    const tip = figure.querySelector(".tip");
    if (!hit || !cross || !tip) return;
    const bandKey = data.band && data.band.indexOf("What-if") === 0 ? "whatif" : "model";

    const row = (label, value, cls) => {
      const line = document.createElement("div");
      line.className = "tip-row";
      const name = document.createElement("span");
      const key = document.createElement("i");
      key.className = "key " + cls;
      name.appendChild(key);
      name.appendChild(document.createTextNode(label));
      const number = document.createElement("b");
      number.textContent = value;
      line.appendChild(name);
      line.appendChild(number);
      return line;
    };

    const show = (clientX) => {
      const box = svg.getBoundingClientRect();
      const scale = data.w / box.width;
      const x = (clientX - box.left) * scale;
      let best = 0;
      data.x.forEach((value, index) => {
        if (Math.abs(value - x) < Math.abs(data.x[best] - x)) best = index;
      });
      const point = data.points[best];
      cross.setAttribute("x1", data.x[best]);
      cross.setAttribute("x2", data.x[best]);
      cross.setAttribute("visibility", "visible");
      tip.replaceChildren();
      const title = document.createElement("div");
      title.className = "tip-title";
      title.textContent = point.label + ("forecast" in point ? " · forecast" : " · actual");
      tip.appendChild(title);
      if ("actual" in point) tip.appendChild(row("Actual", fmt(point.actual), ""));
      if ("forecast" in point) {
        tip.appendChild(row("Forecast", fmt(point.forecast), "forecast"));
        tip.appendChild(row(data.band, fmt(point.low) + " – " + fmt(point.high), bandKey));
        if (point.p10 !== undefined && point.p10 !== null) {
          tip.appendChild(row("Model p10–p90", fmt(point.p10) + " – " + fmt(point.p90), "p"));
        }
      }
      tip.hidden = false;
      const left = data.x[best] / scale;
      tip.style.left = Math.min(Math.max(left, 120), box.width - 120) + "px";
    };
    const hide = () => {
      tip.hidden = true;
      cross.setAttribute("visibility", "hidden");
    };
    hit.addEventListener("pointermove", (event) => show(event.clientX));
    hit.addEventListener("pointerleave", hide);
  });
})();
