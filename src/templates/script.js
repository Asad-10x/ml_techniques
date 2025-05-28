// Form submission
document
  .getElementById("predictForm")
  .addEventListener("predict", async function (e) {
    e.preventDefault();

    const formData = new FormData(e.target);
    const data = {};
    const result = [];
    console.log(formData);
    formData.forEach((value, key) => {
      // Convert numeric fields properly
      if (!isNaN(value)) {
        data[key] = Number(value);
      } else {
        data[key] = value;
      }
    });

    // Validation check (you can improve with custom logic)
    if (
      data["age"] <= 0 ||
      data["educational-num"] <= 0 ||
      data["hours-per-week"] <= 0 ||
      data["capital-gain"] < 0 ||
      data["capital-loss"] < 0
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
        body: JSON.stringify({
          data: {
      "age": formData.age,
      "educational_num": formData.educational_num,
      "capital-gain": formData.capital_gain,
      "capital-loss": formData.capital_loss,
      "hours-per-week": formData.hours_per_week,
      "gender": formData.gender,
      "workclass": formData.workclass,
      "marital_status": formData.marital_status,
      "relationship_type": formData.relationship_type,
     " occupation_type": formData.occupation_type,
      "race": formData.race,
     " native_country": formData.native_country,
    },
        }),
      }).then((res) => res.json())
  .then((data) => {
    console.log(data);
    data.push(data)
    document.getElementById("result").innerText =
        "Prediction: " + res.prediction;
  })
  .catch((err) => {
    console.error(err);
  });
    } catch (error) {
      alert("Error sending data: " + error.message);
    }
  });
