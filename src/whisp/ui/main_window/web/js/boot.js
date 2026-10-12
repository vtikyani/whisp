// Runs before the first paint (a classic, blocking script) so the page never
// flashes the wrong theme. The window passes the theme and motion preference
// in the URL; everything else arrives over the bridge.
(() => {
  const params = new URLSearchParams(location.search);
  const root = document.documentElement;
  root.dataset.theme = params.get("theme") === "light" ? "light" : "dark";
  root.dataset.motion = params.get("motion") === "reduced" ? "reduced" : "full";
})();
