// Pin the Furo theme to light. The smfeval docs are math-heavy and only
// styled for a light background; the theme toggle is hidden in custom.css.
(function () {
  try {
    localStorage.setItem("theme", "light");
  } catch (e) {
    /* localStorage may be unavailable; the data-theme set below still wins */
  }
  function pin() {
    if (document.body) {
      document.body.dataset.theme = "light";
    }
  }
  pin();
  document.addEventListener("DOMContentLoaded", pin);
})();
