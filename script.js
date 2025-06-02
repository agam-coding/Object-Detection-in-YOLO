document.addEventListener("DOMContentLoaded", function () {
  const buttons = ["realtime", "capture", "upload"];
  const sections = {};

  buttons.forEach(id => {
    sections[id] = document.getElementById(id);
    document.getElementById(id + "Btn").addEventListener("click", () => switchMode(id));
  });

  function switchMode(mode) {
    buttons.forEach(id => {
      sections[id].classList.remove("active");
      document.getElementById(id + "Btn").classList.remove("active");
    });
    sections[mode].classList.add("active");
    document.getElementById(mode + "Btn").classList.add("active");
  }

  document.getElementById("clearCanvasBtn").addEventListener("click", () => {
    ["canvas", "captureCanvas", "uploadCanvas"].forEach(id => {
      const canvas = document.getElementById(id);
      if (canvas) {
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
    });
  });

  document.getElementById("downloadImageBtn").addEventListener("click", () => {
    const canvas = document.querySelector(".mode-section.active canvas");
    if (canvas) {
      const link = document.createElement("a");
      link.download = "detection_result.png";
      link.href = canvas.toDataURL();
      link.click();
    }
  });

  let labelsVisible = true;
  document.getElementById("toggleLabelsBtn").addEventListener("click", () => {
    labelsVisible = !labelsVisible;
    alert("Labels " + (labelsVisible ? "shown" : "hidden"));
  });
}); // <-- This closing bracket and parenthesis fix the "Unexpected end of input" error.
