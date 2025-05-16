from flask import Flask, render_template, request, send_file, redirect, url_for
from fpdf import FPDF
import os
import random

import joblib
import numpy as np

app = Flask(__name__)

# 첫 페이지
@app.route('/')
def index():
    return render_template('intro.html')

# 건강 정보 입력 페이지
@app.route('/form')
def form():
    return render_template('form.html')

# 우울증 스코어
def get_melancholia_status(score):

    if score <= 4:
        return "None/Minimal"
    elif score <= 9:
        return "Mild"
    elif score <= 14:
        return "Moderate"
    elif score <= 19:
        return "Moderately Severe"
    else:
        return "Severe"   

# 예측 + 결과 화면 + PDF 리포트 생성
@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()
    
    gender = int(form_data.get("gender", 1))
    age = int(form_data.get("age", 40))
    race = int(form_data.get("race", 5))
    education = int(form_data.get("education", 5))
    income = int(form_data.get("income", 5))
    
    weight = int(form_data.get("weight", 50))
    height = int(form_data.get("height", 170))
    bmi = float(form_data.get("bmi", 23))
    
    sbp1 = int(form_data.get("sbp1", 100))
    dbp1 = int(form_data.get("dbp1", 50))
    sbp2 = int(form_data.get("sbp2", 100))
    dbp2 = int(form_data.get("dbp2", 50))

    smoking = form_data.get("smoking", 2)
    alcohol = form_data.get("alcohol", 2)
    
    '''
    # 예측값 (임시)
    hypertension_risk = round(random.uniform(30, 90), 1)
    diabetes_risk = round(random.uniform(20, 80), 1)
    heart_risk = round(random.uniform(10, 60), 1)
    
    # 추천 운동
    if age > 50 or bmi > 25:
        exercise_recommend = "매일 가벼운 걷기, 요가, 수영을 추천합니다."
    else:
        exercise_recommend = "주 3~5회 유산소 운동과 가벼운 근력운동을 병행하세요."
    
    # 추천 식단
    if smoking == "예" or alcohol == "예":
        diet_recommend = "술과 담배를 줄이고, 해독을 돕는 채소와 물 섭취를 늘리세요."
    elif bmi > 25:
        diet_recommend = "탄수화물을 줄이고, 단백질과 채소 위주의 식단을 권장합니다."
    else:
        diet_recommend = "짜지 않게, 단순당 줄이고, 채소와 단백질 위주로 식사하세요."
    '''
    
    # 1. predict_health_checkup
    # input....
    input_X = [gender, age, race, education, income]
    input_X += [weight, height, bmi]
    input_X += [sbp1, dbp1, sbp2, dbp2]
    input_X += [smoking, alcohol]
    
    clf_diabetes = joblib.load('static/models/diabetes_RF.pkl') 
    output_diabetes = clf_diabetes.predict_proba(np.expand_dims(input_X, axis=0))[0]
    #print(output_diabetes)
    diabetes_result_code = np.argmax(output_diabetes)
    diabetes_result_score = output_diabetes[diabetes_result_code]
    print('diabetes : ', diabetes_result_code + 1, diabetes_result_score)
    # 1 = 예, 2 = 아니오, 3 = Borderline (경계성)

    clf_high_blood_pressure = joblib.load('static/models/high_blood_pressure_RF.pkl') 
    output_high_blood_pressure = clf_high_blood_pressure.predict_proba(np.expand_dims(input_X, axis=0))[0]
    #print(output_high_blood_pressure)
    hb_pressure_result_code = np.argmax(output_high_blood_pressure)
    hb_pressure_result_score = output_high_blood_pressure[hb_pressure_result_code]
    print('high_blood_pressure : ', hb_pressure_result_code + 1, hb_pressure_result_score)    
    # 1	예 (Yes) – 고혈압 진단을 받은 적 있음, 2	아니오 (No) – 고혈압 진단 받은 적 없음
    
    print('BMI (비만) : ', bmi)

    # [임시] 예측 결과 생성 (나중에 실제 모델로 교체)
    hypertension_risk = round(hb_pressure_result_score * 100.0, 2)  #round(random.uniform(30, 90), 1)    # 고혈압
    diabetes_risk = round(diabetes_result_score * 100.0, 2)  #round(random.uniform(20, 80), 1)        # 당뇨...

    # 2.predict_mental_health_checkup

    marital_status = int(form_data.get("marital_status", 1))
    strong_stress = int(form_data.get("strong_stress", 1))
    sleeping_time = int(form_data.get("sleeping_time", 1))
    strong_exercise = int(form_data.get("strong_exercise", 1))
    medium_exercise = int(form_data.get("medium_exercise", 1))
    smoking_01 = int(form_data.get("smoking_01", 1))
    smoking_02 = int(form_data.get("smoking_02", 1))
    alcohol_01 = int(form_data.get("alcohol_01", 1))
    dpq01 = int(form_data.get("dpq01", 1))
    dpq02 = int(form_data.get("dpq02", 1))
    dpq03 = int(form_data.get("dpq03", 1))
    dpq04 = int(form_data.get("dpq04", 1))
    dpq05 = int(form_data.get("dpq05", 1))
    dpq06 = int(form_data.get("dpq06", 1))
    dpq07 = int(form_data.get("dpq07", 1))
    dpq08 = int(form_data.get("dpq08", 1))
    dpq09 = int(form_data.get("dpq09", 1))
    dpq_total = dpq01 + dpq02 + dpq03 + dpq04 + dpq05 + dpq06 + dpq07 + dpq08 + dpq09

    # input....
    input_X = [gender, age, race, education, marital_status, income, strong_stress]
    input_X += [sleeping_time, strong_exercise, medium_exercise, smoking_01, smoking_02, alcohol, alcohol_01, diabetes_result_code + 1, hb_pressure_result_code + 1, bmi]
    input_X += [dpq01, dpq02, dpq03, dpq04, dpq05, dpq06, dpq07, dpq08, dpq09, dpq_total]
    
    clf_mental_health = joblib.load('static/models/mental_health_RF.pkl') 
    output_mental_health = clf_mental_health.predict_proba(np.expand_dims(input_X, axis=0))[0]
    #print(output_mental_health)
    mental_health_result_code = np.argmax(output_mental_health)
    mental_health_result_score = output_mental_health[mental_health_result_code]
    print('mental_health : ', mental_health_result_code + 1, mental_health_result_score)
    #     1	Yes (예, 진단받음), 2	No (아니요, 진단받은 적 없음)
    
    mental_status = get_melancholia_status(dpq_total)
    print('우울증 상태 : ', mental_status)

    mental_health_risk = round(mental_health_result_score * 100.0, 1)
    
    # 3. predict_heart_disease_checkup
    
    smoking_cnt = int(form_data.get("smoking_cnt", 1))
    
    in_drink_freq_per_year_id = int(form_data.get("alcohol_01", 1))
    in_drink_freq_per_year = {1: 365, 2:5.5*52, 3:3.5*52, 4:1.5*52, 5:2.5*12, 6:12, 7:6}

    exercise_days = int(form_data.get("exercise_days", 1))
    bp_had = int(form_data.get("bp_had", 1))
    bp_medication = int(form_data.get("bp_medication", 1)) 
    
    # input....
    input_X = [gender, age, race, education, income]
    input_X += [smoking, smoking_cnt, alcohol, in_drink_freq_per_year[in_drink_freq_per_year_id]]
    input_X += [medium_exercise, strong_exercise, exercise_days, bp_had, bp_medication]
    
    clf_heart_attack = joblib.load('static/models/heart_attack_RF.pkl') 
    output_heart_attack = clf_heart_attack.predict_proba(np.expand_dims(input_X, axis=0))[0]
    #print(output_heart_attack)
    heart_attack_result_code = np.argmax(output_heart_attack)
    heart_attack_result_score = output_heart_attack[heart_attack_result_code]
    print('heart_attack (심장마비) : ', heart_attack_result_code + 1, heart_attack_result_score)
    #     1=Yes, 2=No
    
    clf_angina = joblib.load('static/models/angina_RF.pkl') 
    output_angina = clf_angina.predict_proba(np.expand_dims(input_X, axis=0))[0]
    #print(output_angina)
    angina_result_code = np.argmax(output_angina)
    angina_result_score = output_angina[angina_result_code]
    print('angina (협심증) : ', angina_result_code + 1, angina_result_score)
    #     1=Yes, 2=No   

    heart_attack_risk = round(heart_attack_result_score * 100.0, 2)    
    angina_risk = round(angina_result_score * 100.0, 2)    

    # 추천 로직 (간단한 조건 기반 예시)
    exercise_recommend = "주 3~5회 유산소 운동과 가벼운 근력운동을 병행하세요."
    diet_recommend = "짜지 않게, 단순당 줄이고, 채소와 단백질 위주로 식사하세요."

    # 📄 PDF 생성
    pdf = FPDF()
    pdf.add_page()

    # ✅ 한글 폰트 등록
    font_path = "static/fonts/NotoSansKR-Regular.ttf"
    if not os.path.exists(font_path):
        return "폰트 파일이 존재하지 않습니다. static/fonts/ 폴더에 NotoSansKR-Regular.ttf 추가해주세요."

    pdf.add_font('Noto', '', font_path, uni=True)
    pdf.set_font('Noto', '', 14)

    pdf.cell(0, 10, txt="내몸진단 건강 리포트", ln=True)
    pdf.ln(10)
    pdf.cell(0, 10, txt=f"🩺 예측 질병 위험도", ln=True)
    pdf.cell(0, 10, txt=f"- 고혈압 위험도: {hypertension_risk}%", ln=True)
    pdf.cell(0, 10, txt=f"- 당뇨병 위험도: {diabetes_risk}%", ln=True)
    pdf.cell(0, 10, txt=f"- 비만(BMI): {bmi}", ln=True)
    pdf.cell(0, 10, txt=f"- 우울증 위험도: {mental_health_risk}%", ln=True)
    pdf.cell(0, 10, txt=f"- 심장마비 위험도: {heart_attack_risk}%", ln=True)
    pdf.cell(0, 10, txt=f"- 협심증 위험도: {angina_risk}%", ln=True)
    pdf.ln(10)
    pdf.multi_cell(0, 10, f"🏃 추천 운동:\n{exercise_recommend}")
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"🥗 추천 식단:\n{diet_recommend}")

    # 저장
    pdf.output("static/reports/report.pdf")

    return render_template(
        'report.html',
        hypertension_risk=hypertension_risk,
        diabetes_risk=diabetes_risk,
        bmi=bmi,
        mental_health_risk=mental_health_risk,
        heart_attack_risk=heart_attack_risk,
        angina_risk=angina_risk,
        exercise_recommend=exercise_recommend,
        diet_recommend=diet_recommend
    )

# PDF 다운로드
@app.route('/download_report')
def download_report():
    return send_file("static/reports/report.pdf", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)