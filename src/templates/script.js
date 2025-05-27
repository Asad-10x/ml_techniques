// Form submission
document
  .getElementById("predictForm")
  .addEventListener("predict", async function (e) {
    e.preventDefault();

    const formData = new FormData(e.target);
    const jsonData = {};
    console.log(formData);
    formData.forEach((value, key) => {
      // Convert numeric fields properly
      if (!isNaN(value)) {
        jsonData[key] = Number(value);
      } else {
        jsonData[key] = value;
      }
    });

    // Validation check (you can improve with custom logic)
    if (
      jsonData["age"] <= 0 ||
      jsonData["educational-num"] <= 0 ||
      jsonData["hours-per-week"] <= 0 ||
      jsonData["capital-gain"] < 0 ||
      jsonData["capital-loss"] < 0
    ) {
      alert("Please enter valid positive values for numeric fields.");
      return;
    }
    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(jsonData),
      });

      const result = await response.json();
      document.getElementById("result").innerText =
        "Prediction: " + result.prediction;
    } catch (error) {
      alert("Error sending data: " + error.message);
    }
  });
