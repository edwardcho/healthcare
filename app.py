from flask import Flask, render_template, request, send_file, redirect, url_for, session, after_this_request
from fpdf import FPDF
import os
import random

import joblib
import numpy as np

import uuid
import json

import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = '1596716541.5910525'

# 첫 페이지
@app.route('/')
def index():
    return render_template('intro.html')

'''
# 건강 정보 입력 페이지
@app.route('/form')
def form():
    return render_template('form.html')
'''

@app.route('/health', methods=['GET', 'POST'])
def health():
    return render_template('health_info.html')

@app.route('/mental', methods=['POST'])
def mental():
    session['health_data'] = request.form.to_dict()
    return render_template('mental_health.html')

@app.route('/heart', methods=['POST'])
def heart():
    session['mental_data'] = request.form.to_dict()
    return render_template('heart_health.html')

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
    
    #final_data = {**session.get('health_data', {}), **session.get('mental_data', {}), **heart_data}
    #print(final_data)
    #exit(0)
    
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
    #hypertension_risk = round(hb_pressure_result_score * 100.0, 2)  #round(random.uniform(30, 90), 1)    # 고혈압
    #diabetes_risk = round(diabetes_result_score * 100.0, 2)  #round(random.uniform(20, 80), 1)        # 당뇨...
    hypertension_risk = round(output_high_blood_pressure[0] * 100.0, 2)  #round(random.uniform(30, 90), 1)    # 고혈압
    diabetes_risk = round(output_diabetes[0] * 100.0, 2)  #round(random.uniform(20, 80), 1)        # 당뇨...

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

    #mental_health_risk = round(mental_health_result_score * 100.0, 1)
    mental_health_risk = round(output_mental_health[0] * 100.0, 1)
    
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

    #heart_attack_risk = round(heart_attack_result_score * 100.0, 2)    
    #angina_risk = round(angina_result_score * 100.0, 2)    
    heart_attack_risk = round(output_heart_attack[0] * 100.0, 2)    
    angina_risk = round(output_angina[0] * 100.0, 2)    

    # 추천 로직 (간단한 조건 기반 예시)
    #exercise_recommend = "주 3~5회 유산소 운동과 가벼운 근력운동을 병행하세요."
    #diet_recommend = "짜지 않게, 단순당 줄이고, 채소와 단백질 위주로 식사하세요."
    
    exercise_recommend = ''
    diet_recommend = ''

    hb_pressure_code = "정상"
    if hb_pressure_result_code + 1 == 1:
        hb_pressure_code = "위험"
        exercise_recommend = '혈압을 낮추기 위해 매일 30분 이상 빠르게 걷기나 수영 같은 유산소 운동을 실천해보세요.'
        diet_recommend = '나트륨 섭취를 줄이고, 과일과 채소가 풍부한 DASH 식단을 실천하세요.'
        
    elif hb_pressure_result_code + 1 == 2:
        hb_pressure_code = "정상"
        exercise_recommend = '혈압 유지를 위해 규칙적인 유산소 운동을 계속 유지하세요.'
        diet_recommend = '싱겁게 먹는 습관을 유지하고, 채소와 과일 섭취를 꾸준히 하세요.'
    
    diabetes_code = "정상"
    if diabetes_result_code + 1 == 1:
        diabetes_code = "위험"
        exercise_recommend += '\n' + '당 조절을 위해 매일 유산소 운동과 주 2회 이상 근력 운동을 병행하세요.'
        diet_recommend += '\n' + '정제 탄수화물을 피하고, 식이섬유가 풍부한 저지방 식품 위주로 식사하세요.'
        
    elif diabetes_result_code + 1 == 2:
        diabetes_code = "정상"
        exercise_recommend += '\n' + '당뇨 예방을 위해 걷기, 자전거 타기 같은 유산소 운동을 꾸준히 하세요.'
        diet_recommend += '\n' + '혈당 안정에 도움이 되는 통곡물과 채소 중심 식단을 유지하세요.'

    elif diabetes_result_code + 1 == 3:
        diabetes_code = "가능성있음"
        exercise_recommend += '\n' + '당 조절을 위해 매일 유산소 운동과 주 2회 이상 근력 운동을 병행하세요.'
        diet_recommend += '\n' + '정제 탄수화물을 피하고, 식이섬유가 풍부한 저지방 식품 위주로 식사하세요.'
    
    mental_health_code = '정상'    
    if mental_health_result_code + 1 == 1:
        mental_health_code = '위험'
        exercise_recommend += '\n' + '기분 개선을 위해 산책, 요가, 가벼운 스트레칭을 하루 20~30분 실천해보세요.'
        diet_recommend += '\n' + '오메가-3가 풍부한 음식과 비타민B가 많은 채소, 견과류를 섭취해보세요.'
        
    elif mental_health_result_code + 1 == 2:
        mental_health_code = '정상' 
        exercise_recommend += '\n' + '정신 건강 유지를 위해 가벼운 신체활동을 규칙적으로 유지하세요.'
        diet_recommend += '\n' + '기분 안정을 돕는 생선, 과일, 채소 위주 식단을 계속 유지해보세요.'

    heart_attack_code = '정상'
    if heart_attack_result_code + 1 == 1:        
        heart_attack_code = '위험'
        exercise_recommend += '\n' + '의사 상담 후 가벼운 걷기부터 시작해 점차 활동량을 늘리세요. 과도한 무리 운동은 피하세요.'
        diet_recommend += '\n' + '콜레스테롤과 포화지방이 적은 식사를 하고, 심장 건강에 좋은 생선, 견과류를 섭취하세요.'
        
    elif heart_attack_result_code + 1 == 2:        
        heart_attack_code = '정상'
        exercise_recommend += '\n' + '심혈관 건강 유지를 위해 주 3~5회 유산소 중심 운동을 계속하세요.'
        diet_recommend += '\n' + '심장을 보호하는 지중해식 식단을 유지하세요. 좋은 지방을 포함하는 식품이 도움이 됩니다.'
        
    angina_code = '정상'
    if angina_result_code + 1 == 1:
        angina_code = '위험'
        exercise_recommend += '\n' + '가벼운 산책부터 시작해 점차 활동량을 늘리되, 무리하지 않도록 조절하세요.'
        diet_recommend += '\n' + '지방과 염분이 낮은 식단을 실천하고, 심혈관 보호에 좋은 식품을 선택하세요.'

    elif angina_result_code + 1 == 2:
        angina_code = '정상'
        exercise_recommend += '\n' + '심장 건강을 위해 걷기 등 저강도 유산소 운동을 꾸준히 유지하세요.'
        diet_recommend += '\n' + '심혈관 예방을 위한 저염, 저지방 중심 식단을 유지하세요.'
        
    session['used_uuid'] = f"{uuid.uuid4().hex}"    
    
    used_datas = {}
    used_datas['hb_pressure_code'] = hb_pressure_code
    used_datas['hypertension_risk'] = hypertension_risk
    used_datas['diabetes_code'] = diabetes_code
    used_datas['diabetes_risk'] = diabetes_risk
    used_datas['bmi'] = bmi
    used_datas['mental_health_code'] = mental_health_code
    used_datas['mental_status'] = mental_status
    used_datas['mental_health_risk'] = mental_health_risk
    used_datas['heart_attack_code'] = heart_attack_code
    used_datas['heart_attack_risk'] = heart_attack_risk
    used_datas['angina_code'] = angina_code
    used_datas['angina_risk'] = angina_risk
    used_datas['exercise_recommend'] = exercise_recommend
    used_datas['diet_recommend'] = diet_recommend
    
    with open(os.path.join("static/reports", "used_" + session['used_uuid'] + ".json"), 'w') as f:
        json.dump(used_datas, f)

    return render_template(
        'report.html',
        hb_pressure_code=hb_pressure_code,
        hypertension_risk=hypertension_risk,
        diabetes_code=diabetes_code,
        diabetes_risk=diabetes_risk,
        bmi=bmi,
        mental_health_code=mental_health_code,
        mental_status=mental_status,
        mental_health_risk=mental_health_risk,
        heart_attack_code=heart_attack_code,
        heart_attack_risk=heart_attack_risk,
        angina_code=angina_code,
        angina_risk=angina_risk,
        exercise_recommend=exercise_recommend,
        diet_recommend=diet_recommend
    )

def gen_pdf_report():

    with open(os.path.join("static/reports", "used_" + session['used_uuid'] + ".json"), 'r') as f:
        used_datas = json.load(f)

    hb_pressure_code = used_datas['hb_pressure_code']
    hypertension_risk = used_datas['hypertension_risk']
    diabetes_code = used_datas['diabetes_code']
    diabetes_risk = used_datas['diabetes_risk']
    bmi = used_datas['bmi']
    mental_health_code = used_datas['mental_health_code']
    mental_status = used_datas['mental_status']
    mental_health_risk = used_datas['mental_health_risk']
    heart_attack_code = used_datas['heart_attack_code']
    heart_attack_risk = used_datas['heart_attack_risk']
    angina_code = used_datas['angina_code']
    angina_risk = used_datas['angina_risk']
    exercise_recommend = used_datas['exercise_recommend']
    diet_recommend = used_datas['diet_recommend']
    
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
    pdf.cell(0, 10, txt=f"- 고혈압 : {hb_pressure_code} (위험도: {hypertension_risk}%)", ln=True)
    pdf.cell(0, 10, txt=f"- 당뇨병 : {diabetes_code} (위험도: {diabetes_risk}%)", ln=True)
    pdf.cell(0, 10, txt=f"- 비만(BMI): {bmi}", ln=True)
    pdf.cell(0, 10, txt=f"- 우울증 : {mental_health_code} ({mental_status}) (위험도: {mental_health_risk}%)", ln=True)
    pdf.cell(0, 10, txt=f"- 심장마비 : {heart_attack_code} (위험도: {heart_attack_risk}%)", ln=True)
    pdf.cell(0, 10, txt=f"- 협심증 : {angina_code} (위험도: {angina_risk}%)", ln=True)
    pdf.ln(10)
    pdf.multi_cell(0, 10, f"🏃 추천 운동:\n{exercise_recommend}")
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"🥗 추천 식단:\n{diet_recommend}")

    # 저장
    pdf.output(os.path.join("static/reports", "report_" + session['used_uuid'] + ".pdf"))

# PDF 다운로드
@app.route('/download_report')
def download_report():

    gen_pdf_report()
    send_pdf_filepath = os.path.join("static/reports", "report_" + session['used_uuid'] + ".pdf")
    
    @after_this_request
    def remove_file(response):
        try:
            os.remove(send_pdf_filepath)
        except Exception as e:
            app.logger.error(f"파일 삭제 실패: {e}")
        return response    
    
    return send_file(send_pdf_filepath, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)