
/* ==================================================
   AQUAWATCH - FRONTEND SCRIPT
   Flood Detection + ML Prediction
================================================== */

const API_URL = "http://127.0.0.1:8000";

const MAX_FILE_SIZE = 15 * 1024 * 1024;

const ALLOWED_TYPES = [
    "image/jpeg",
    "image/png",
    "image/webp"
];


/* ==================================================
   1. COMMON API RESPONSE HANDLER
================================================== */

async function handleApiResponse(response) {

    console.log("API Status:", response.status);

    let responseText = "";

    try {
        responseText = await response.text();
    } catch (error) {
        throw new Error("Unable to read backend response.");
    }

    console.log("API Response:", responseText);

    if (!responseText || !responseText.trim()) {

        throw new Error(
            `Backend returned an empty response (HTTP ${response.status}).`
        );
    }

    let result;

    try {

        result = JSON.parse(responseText);

    } catch (error) {

        throw new Error(
            "Backend did not return valid JSON. Check FastAPI terminal."
        );

    }

    if (!response.ok) {

        let errorMessage = "Backend request failed.";

        if (result.detail) {

            if (Array.isArray(result.detail)) {

                errorMessage = result.detail
                    .map(item => item.msg || JSON.stringify(item))
                    .join(", ");

            } else {

                errorMessage = result.detail;

            }

        } else if (result.message) {

            errorMessage = result.message;

        }

        throw new Error(
            `HTTP ${response.status}: ${errorMessage}`
        );
    }

    return result;
}


/* ==================================================
   2. IMAGE VALIDATION
================================================== */

function validateImage(file) {

    if (!file) {

        throw new Error(
            "Please select both satellite images."
        );

    }

    if (!ALLOWED_TYPES.includes(file.type)) {

        throw new Error(
            "Only JPG, PNG and WEBP images are allowed."
        );

    }

    if (file.size > MAX_FILE_SIZE) {

        throw new Error(
            "Image size must not exceed 15 MB."
        );

    }

    if (file.size === 0) {

        throw new Error("Selected image is empty.");

    }

    return true;
}


/* ==================================================
   3. IMAGE PREVIEW
================================================== */

function previewImage(inputId, previewId) {

    const input = document.getElementById(inputId);

    const preview = document.getElementById(previewId);

    if (!input || !preview) {

        console.warn(`Missing image elements: ${inputId}`);

        return;
    }

    input.addEventListener("change", function () {

        const file = this.files[0];

        if (!file) {

            preview.removeAttribute("src");

            preview.style.display = "none";

            return;
        }

        try {

            validateImage(file);

            const reader = new FileReader();

            reader.onload = function (event) {

                preview.src = event.target.result;

                preview.style.display = "block";

            };

            reader.onerror = function () {

                showPredictionMessage(
                    "Unable to read selected image.",
                    true
                );

            };

            reader.readAsDataURL(file);

        } catch (error) {

            this.value = "";

            preview.removeAttribute("src");

            preview.style.display = "none";

            showPredictionMessage(
                error.message,
                true
            );

        }

    });

}


/* ==================================================
   4. PREDICTION MESSAGE
================================================== */

function showPredictionMessage(message, isError = false) {

    const element = document.getElementById(
        "predictionMessage"
    );

    if (!element) {

        console.log(message);

        return;
    }

    element.innerText = message;

    element.style.color = isError
        ? "#dc2626"
        : "#16a34a";

}


/* ==================================================
   5. EXISTING FLOOD DETECTION
================================================== */

async function runDetection() {

    const button = document.getElementById("detectBtn");

    const status = document.getElementById("statusMessage");

    if (!button || !status) {

        console.error(
            "Flood detection elements not found."
        );

        return;
    }

    button.disabled = true;

    button.innerText = "Processing...";

    status.innerText = "Analyzing flood data...";

    try {

        const response = await fetch(
            `${API_URL}/detect-flood`,
            {
                method: "POST"
            }
        );

        const result = await handleApiResponse(response);

        const data = result.data || result;

        console.log(
            "Flood Detection Result:",
            data
        );

        // Flood area
        const floodArea = document.getElementById(
            "floodArea"
        );

        if (floodArea) {
            floodArea.innerText =
                data.flood_area ?? "--";
        }

        // Flood pixels
        const floodPixels = document.getElementById(
            "floodPixels"
        );

        if (floodPixels) {
            floodPixels.innerText =
                data.flood_pixels ?? "--";
        }

        // Threshold
        const threshold = document.getElementById(
            "threshold"
        );

        if (threshold) {
            threshold.innerText =
                data.threshold ?? "--";
        }

        // Villages
        const villages = document.getElementById(
            "villages"
        );

        if (villages) {
            villages.innerText =
                data.villages ?? "--";
        }

        // Comparison image
        const comparisonImage = document.getElementById(
            "comparisonImage"
        );

        if (comparisonImage) {

            comparisonImage.src =
                `${API_URL}/reports/flood_comparison.png?t=${Date.now()}`;

            comparisonImage.style.display = "block";

        }

        // Flood map
        const floodMap = document.getElementById(
            "floodMap"
        );

        if (floodMap) {

            floodMap.src =
                `${API_URL}/reports/flood_map.html?t=${Date.now()}`;

            floodMap.style.display = "block";

        }

        status.innerText =
            result.message ||
            "Flood detection completed successfully!";

    } catch (error) {

        console.error(
            "Flood Detection Error:",
            error
        );

        status.innerText = error.message;

    } finally {

        button.disabled = false;

        button.innerText = "Detect Flood";

    }

}


/* ==================================================
   6. ML FLOOD PREDICTION
================================================== */

async function predictFlood() {

    const beforeInput = document.getElementById(
        "beforeImage"
    );

    const afterInput = document.getElementById(
        "afterImage"
    );

    const button = document.getElementById(
        "predictBtn"
    );

    const results = document.getElementById(
        "predictionResults"
    );

    if (!beforeInput || !afterInput || !button) {

        showPredictionMessage(
            "Prediction HTML elements are missing.",
            true
        );

        return;
    }

    const beforeFile = beforeInput.files[0];

    const afterFile = afterInput.files[0];

    try {

        // Validate images
        validateImage(beforeFile);

        validateImage(afterFile);

        // Disable button
        button.disabled = true;

        button.innerText = "Analyzing Images...";

        showPredictionMessage(
            "AI model is analyzing satellite images. Please wait..."
        );

        if (results) {
            results.style.display = "none";
        }

        // Prepare form data
        const formData = new FormData();

        formData.append(
            "before_image",
            beforeFile
        );

        formData.append(
            "after_image",
            afterFile
        );

        console.log(
            "Uploading images to AquaWatch backend..."
        );

        console.log("Before image:", beforeFile.name);

        console.log("After image:", afterFile.name);

        // Send request
        let response;

        try {

            response = await fetch(
                `${API_URL}/predict-flood`,
                {
                    method: "POST",
                    body: formData
                }
            );

        } catch (networkError) {

            throw new Error(
                "Cannot connect to AquaWatch backend. Check whether FastAPI is running."
            );

        }

        // Process backend response
        const result = await handleApiResponse(response);

        const data = result.data || result;

        console.log(
            "ML Prediction Result:",
            data
        );


        // ------------------------------------------
        // FLOOD PROBABILITY HANDLING
        // ------------------------------------------

        let probability = Number(
            data.flood_probability ??
            data.probability ??
            data.flood_percentage
        );

        if (!Number.isFinite(probability)) {

            console.error(
                "Unexpected ML response:",
                data
            );

            throw new Error(
                "Flood probability is missing from the backend response."
            );

        }

        // Support percentage values such as 85
        if (probability > 1 && probability <= 100) {

            probability = probability / 100;

        }

        if (probability < 0 || probability > 1) {

            throw new Error(
                "Invalid flood probability received from the ML model."
            );

        }

        const percentage = probability * 100;

        console.log(
            "Flood Probability:",
            percentage.toFixed(2) + "%"
        );


        // ------------------------------------------
        // PREDICTION STATUS
        // ------------------------------------------

        const predictionStatus = document.getElementById(
            "predictionStatus"
        );

        if (predictionStatus) {

            if (data.flooded === true) {

                predictionStatus.innerText =
                    "Flood Detected";

            } else if (data.flooded === false) {

                predictionStatus.innerText =
                    "No Flood Detected";

            } else {

                predictionStatus.innerText =
                    "Prediction Completed";

            }

        }


        // ------------------------------------------
        // PROBABILITY DISPLAY
        // ------------------------------------------

        const probabilityElement = document.getElementById(
            "floodProbability"
        );

        if (probabilityElement) {

            probabilityElement.innerText =
                `${percentage.toFixed(2)}%`;

        }


        // ------------------------------------------
        // PROBABILITY PROGRESS BAR
        // ------------------------------------------

        const progressBar = document.getElementById(
            "probabilityBar"
        );

        if (progressBar) {

            progressBar.style.width =
                `${percentage}%`;

            progressBar.setAttribute(
                "aria-valuenow",
                percentage.toFixed(2)
            );

        }


        // ------------------------------------------
        // DEVICE INFORMATION
        // ------------------------------------------

        const deviceElement = document.getElementById(
            "predictionDevice"
        );

        if (deviceElement) {

            deviceElement.innerText =
                data.device || "CPU";

        }


        // ------------------------------------------
        // PREDICTION NOTE
        // ------------------------------------------

        const noteElement = document.getElementById(
            "predictionNote"
        );

        if (noteElement) {

            noteElement.innerText =
                data.prediction_note ||
                "ML prediction completed successfully.";

        }


        // ------------------------------------------
        // SHOW RESULTS
        // ------------------------------------------

        if (results) {

            results.style.display = "block";

        }

        showPredictionMessage(
            result.message ||
            "Flood prediction completed successfully!"
        );

    } catch (error) {

        console.error(
            "ML Prediction Error:",
            error
        );

        showPredictionMessage(
            `Prediction error: ${error.message}`,
            true
        );

    } finally {

        button.disabled = false;

        button.innerText = "Predict Flood";

    }

}


/* ==================================================
   7. INITIALIZE AQUAWATCH
================================================== */

document.addEventListener(
    "DOMContentLoaded",
    function () {

        console.log(
            "AquaWatch frontend initialized successfully."
        );

        // Before image preview
        previewImage(
            "beforeImage",
            "beforePreview"
        );

        // After image preview
        previewImage(
            "afterImage",
            "afterPreview"
        );

        // Hide previews initially
        const beforePreview = document.getElementById(
            "beforePreview"
        );

        const afterPreview = document.getElementById(
            "afterPreview"
        );

        if (beforePreview) {

            beforePreview.style.display = "none";

        }

        if (afterPreview) {

            afterPreview.style.display = "none";

        }

        // Flood detection button
        const detectButton = document.getElementById(
            "detectBtn"
        );

        if (detectButton) {

            detectButton.addEventListener(
                "click",
                runDetection
            );

        }

        // ML prediction button
        const predictButton = document.getElementById(
            "predictBtn"
        );

        if (predictButton) {

            predictButton.addEventListener(
                "click",
                predictFlood
            );

        }

    }
);
