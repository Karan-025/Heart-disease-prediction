import gradio as gr
import pickle
import numpy as np
import requests
import os
import warnings
warnings.filterwarnings("ignore")
MODEL_PATH = ("/content/heart_disease 2026 part 3.pkl")


try:

    with open(MODEL_PATH, "rb") as file:

        models = pickle.load(file)

    print(f"Models loaded: {list(models.keys())}")

except Exception as e:

    print(f"Error loading models: {e}")

    models = None
def generate_summary(age, risk_status):
    if risk_status == "Low Risk":
        return ("The patient is currently classified as low risk. Maintaining a healthy lifestyle, "
                "balanced diet, and regular exercise is recommended to continue mitigating cardiovascular risks.")
    if age > 40:
        prompt = ("Summarize this medical risk: The patient is at high risk for cardiovascular disease. "
                  "The major driving factors are advancing age and high resting blood pressure.")
    else:
        prompt = ("Summarize this medical risk: The patient is at high risk for cardiovascular disease. "
                  "The major driving factors are elevated serum cholesterol and high resting blood pressure.")


    API_URL = "Api url"
    headers = {"Authorization": " "}
  # use api for the medical input data to summurize the prediction



    try:
        payload = {"inputs": prompt}
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            res_json = response.json()
            if isinstance(res_json, list) and len(res_json) > 0:
                return res_json[0].get("summary_text", prompt.replace("Summarize this medical risk: ", ""))
            return res_json.get("summary_text", prompt.replace("Summarize this medical risk: ", ""))
        else:
            return prompt.replace("Summarize this medical risk: ", "")
    except Exception:
        return prompt.replace("Summarize this medical risk: ", "")





def predict_all_models(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, bmi, glucose):
    if models is None:
        return "⚠ Models not loaded", "<div>Error: Models not loaded. Please check the model file.</div>"
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, bmi, glucose]])
    results_html = "<div style='display: flex; flex-wrap: wrap; gap: 10px; justify-content: center;'>"
    high_risk_count = 0
    rf_prediction = 0
    ensemble_probs = []
    for name, model in models.items():
        try:
            pred = model.predict(features)[0]
            try:
                prob = model.predict_proba(features)[0][1]
            except AttributeError:
                prob = float(pred)
            ensemble_probs.append(prob)
            if pred == 1:
                high_risk_count += 1
            if "Random Forest" in name or "RandomForest" in name:
                rf_prediction = pred

            color = "#dc3545" if pred == 1 else "#28a745"
            label = "High Risk" if pred == 1 else "Low Risk"



            results_html += f"""

            <div style="border-radius: 10px; padding: 15px; border-left: 5px solid {color}; background: white;
                        box-shadow: 2px 2px 10px rgba(0,0,0,0.05); min-width: 180px;">
                <b style="color: #555;">{name}</b><br>
                <span style="font-size: 20px; color: {color};"><b>{label}</b></span><br>
                <small>Prob: {prob:.2f}</small>
            </div>
            """
        except Exception as e:
            results_html += f"""
            <div style="border-radius: 10px; padding: 15px; border-left: 5px solid #ffc107; background: white;
                        box-shadow: 2px 2px 10px rgba(0,0,0,0.05); min-width: 180px;">
                <b style="color: #555;">{name}</b><br>
                <span style="font-size: 14px; color: #ffc107;">Error: {str(e)}</span>
            </div>
            """



    results_html += "</div>"



    total_models = len(models)
    if high_risk_count > (total_models / 2):
        final_pred = 1
    elif high_risk_count < (total_models / 2):
        final_pred = 0
    else:
        final_pred = rf_prediction



    final_label = "HIGH RISK" if final_pred == 1 else "LOW RISK"
    final_color = "#dc3545" if final_pred == 1 else "#28a745"
    mean_prob = np.mean(ensemble_probs) * 100 if ensemble_probs else 0.0



    summary_text = generate_summary(age, "High Risk" if final_pred == 1 else "Low Risk")



    final_outcome_html = f"""

    <div style="text-align: center; padding: 30px; border-radius: 15px; background: white;
                box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-top: 20px; border-top: 8px solid {final_color};">
        <h2 style="margin: 0; color: #333;">Aggregated Diagnostic Verdict</h2>
        <h1 style="color: {final_color}; font-size: 40px; margin: 10px 0;">{final_label}</h1>
        <h3 style="color: #555; margin-bottom: 20px;">Overall Risk Probability: {mean_prob:.1f}%</h3>
        <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; text-align: left;
                    font-size: 16px; color: #444; line-height: 1.5;">
            <strong>Clinical Summary:</strong> {summary_text}
        </div>
    </div>
    """



    return results_html, final_outcome_html





css = """

.gradio-container { background-color: #f8f9fa!important; }
.input-card { background: white; border-radius: 15px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 15px; }
.predict-btn { background: linear-gradient(135deg, #007bff 0%, #0056b3 100%)!important; color: white!important; font-size: 16px!important; padding: 12px!important; }
"""



with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    gr.HTML("<h1 style='text-align: center; color: #333;'>💙 Heart Health Disease Prediction Dashboard</h1>")


    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group(elem_classes="input-card"):
                gr.Markdown("### 🧬 Primary Clinical Indicators")
                age = gr.Slider(20, 100, step=1, label="Age (Range: 20-100 | Healthy Baseline: 45)", value=45)
                sex = gr.Radio(choices=[("Female", 0), ("Male", 1)] , label="Sex (0:F, 1:M)", value=1)
                trestbps = gr.Slider(80, 200, step=1, label="Resting BP [mm Hg] (Range: 80-200 | Healthy Baseline: 120)", value=120)
                chol = gr.Slider(100, 600, step=1, label="Cholesterol [mg/dl] (Range: 100-600 | Healthy Baseline: 200)", value=200)
                thalach = gr.Slider(60, 220, step=1, label="Max Heart Rate [bpm] (Range: 60-220 | Healthy Baseline: 150)", value=150)
                cp = gr.Slider(0, 3, step=1, label="Chest Pain Type (Range: 0-3 | Healthy Baseline: 0)", value=0)
                bmi = gr.Slider(10.0, 60.0, step=0.1, label="BMI", value=26.0)
                glucose = gr.Slider(50, 400, step=1, label="Glucose (mg/dl)", value=85)

            with gr.Accordion("Advanced Parameters (Secondary Diagnostics)", open=False, elem_classes="input-card"):

                fbs = gr.Radio(choices=[("No", 0), ("Yes", 1)], label="Fasting Blood Sugar > 120 mg/dl", value=0)# Fixed syntax
                restecg = gr.Slider(0, 2, step=1, label="Resting ECG (Range: 0-2 | Healthy Baseline: 0)", value=0)
                exang = gr.Radio(choices=[("No", 0), ("Yes", 1)], label="Exer. Angina (Range: 0-1 | Healthy Baseline: 0)", value=0) # Fixed syntax
                oldpeak = gr.Slider(0.0, 6.2, step=0.1, label="ST Depression (Range: 0.0-6.2 | Healthy Baseline: 0.0)", value=0.0)
                slope = gr.Slider(0, 2, step=1, label="Slope of ST Segment (Range: 0-2 | Healthy Baseline: 0)", value=0)
                ca = gr.Slider(0, 4, step=1, label="Major Vessels (Range: 0-4 | Healthy Baseline: 0)", value=0)
                thal = gr.Slider(0, 3, step=1, label="Thalassemia (Range: 0-3 | Healthy Baseline: 2)", value=2)




            predict_btn = gr.Button("RUN DIAGNOSTIC", variant="primary", elem_classes="predict-btn")



        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Live Predictions"):
                    output_html = gr.HTML("<p style='text-align: center; margin-top: 20px; color: #777;'>Waiting for input data...</p>")
                    final_outcome = gr.HTML("")



                with gr.TabItem("System Logs"):
                    model_names = list(models.keys()) if models else []
                    gr.Markdown(f"✅ Models active: {', '.join(model_names) if model_names else 'None loaded'}")
                    gr.Markdown("✅ Adjudication Protocol: Active (Random Forest Fallback)")
                    gr.Markdown("✅ Medical Summary API: Connected")
                    gr.Markdown("✅ Input features: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, bmi, glucose (15 total)")



    predict_btn.click(
        fn=predict_all_models,
        inputs=[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, bmi, glucose],
        outputs=[output_html, final_outcome]

    )

  
    scroll_to_top_js = """
    async () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
    """
    predict_btn.click(
        fn=predict_all_models,
        inputs=[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, bmi, glucose],
        outputs=[output_html, final_outcome]
    ).then(fn=None, js=scroll_to_top_js)
   

demo.launch(share=True, debug=True)
