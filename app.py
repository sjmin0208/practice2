import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import streamlit.components.v1 as components
import json

st.set_page_config(
    page_title="증상 기반 질병·인체 시각화 대시보드",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.warning("⚠️ **면책 조항**: 이 도구는 의료기기가 아니며 진단·처방을 대체하지 않습니다. 약품 복용 전 반드시 의사·약사와 상담하세요.")

# ════════════════════════════════════════════════════════
#  데이터
# ════════════════════════════════════════════════════════
DISEASE_KR = {
    "Fungal infection":"곰팡이 감염","Allergy":"알레르기","GERD":"위식도역류",
    "Chronic cholestasis":"만성 담즙정체","Drug Reaction":"약물 반응",
    "Peptic ulcer disease":"소화성 궤양","AIDS":"에이즈","Diabetes":"당뇨",
    "Gastroenteritis":"위장염","Bronchial Asthma":"기관지 천식",
    "Hypertension":"고혈압","Migraine":"편두통","Cervical spondylosis":"경추 척추증",
    "Paralysis (brain hemorrhage)":"뇌출혈/마비","Jaundice":"황달",
    "Malaria":"말라리아","Chicken pox":"수두","Dengue":"뎅기열",
    "Typhoid":"장티푸스","hepatitis A":"A형 간염","Hepatitis B":"B형 간염",
    "Hepatitis C":"C형 간염","Hepatitis D":"D형 간염","Hepatitis E":"E형 간염",
    "Alcoholic hepatitis":"알코올성 간염","Tuberculosis":"결핵",
    "Common Cold":"감기","Pneumonia":"폐렴",
    "Dimorphic hemorrhoids(piles)":"치질","Heart attack":"심근경색",
    "Varicose veins":"정맥류","Hypothyroidism":"갑상선 기능 저하",
    "Hyperthyroidism":"갑상선 기능 항진","Hypoglycemia":"저혈당",
    "Osteoarthritis":"골관절염","Arthritis":"관절염",
    "(vertigo) Paroxysmal Positional Vertigo":"이석증(어지럼증)",
    "Acne":"여드름","Urinary tract infection":"요로 감염",
    "Psoriasis":"건선","Impetigo":"농가진",
}

SYMPTOM_KR = {
    "itching":"가려움증","skin_rash":"피부 발진","nodal_skin_eruptions":"결절성 피부 발진",
    "continuous_sneezing":"지속적 재채기","shivering":"떨림","chills":"오한",
    "joint_pain":"관절통","stomach_pain":"복통","acidity":"위산 과다",
    "ulcers_on_tongue":"구내염","muscle_wasting":"근육 소모","vomiting":"구토",
    "burning_micturition":"배뇨 시 화끈감","spotting_urination":"혈뇨",
    "fatigue":"피로감","weight_gain":"체중 증가","anxiety":"불안감",
    "cold_hands_and_feets":"손발 냉증","mood_swings":"감정 기복",
    "weight_loss":"체중 감소","restlessness":"안절부절","lethargy":"무기력증",
    "patches_in_throat":"인후 반점","irregular_sugar_level":"혈당 불규칙",
    "cough":"기침","high_fever":"고열","sunken_eyes":"움푹 꺼진 눈",
    "breathlessness":"호흡 곤란","sweating":"발한(땀)","dehydration":"탈수",
    "indigestion":"소화불량","headache":"두통","yellowish_skin":"황달(피부)",
    "dark_urine":"짙은 소변","nausea":"메스꺼움","loss_of_appetite":"식욕 부진",
    "pain_behind_the_eyes":"눈 뒤 통증","back_pain":"허리 통증",
    "constipation":"변비","abdominal_pain":"복부 통증","diarrhoea":"설사",
    "mild_fever":"미열","yellow_urine":"황색 소변","yellowing_of_eyes":"눈 황달",
    "acute_liver_failure":"급성 간부전","fluid_overload":"체액 과다",
    "swelling_of_stomach":"복부 팽만","swelled_lymph_nodes":"림프절 부종",
    "malaise":"전신 불쾌감","blurred_and_distorted_vision":"시야 흐림",
    "phlegm":"가래","throat_irritation":"인후 자극","redness_of_eyes":"눈 충혈",
    "sinus_pressure":"부비동 압박","runny_nose":"콧물","congestion":"코막힘",
    "chest_pain":"흉통","weakness_in_limbs":"사지 무력감",
    "fast_heart_rate":"빠른 심박수","pain_during_bowel_movements":"배변 통증",
}

BODY_PART_KR = {
    "brain":"뇌","heart":"심장","lungs":"폐","liver":"간","stomach":"위",
    "intestine":"장","kidney":"신장","skin":"피부","joints":"관절",
    "thyroid":"갑상선","pancreas":"췌장","lymph":"림프계","blood":"혈액",
    "spine":"척추","eye":"눈","nose":"코","neck":"경추","esophagus":"식도",
    "gallbladder":"담낭","spleen":"비장","bladder":"방광","ear":"귀",
    "legs":"하지/정맥","immune":"면역계",
}

BODY_PART_DESC = {
    "brain":"뇌는 중추신경계의 핵심 기관입니다. 편두통, 뇌출혈, 이석증 등의 질환에서 직접 손상이나 기능 이상이 발생합니다.",
    "heart":"심장은 혈액 순환의 펌프입니다. 심근경색은 관상동맥 폐색으로 심근이 괴사하는 응급 상황으로 즉각적인 치료가 필요합니다.",
    "lungs":"폐는 산소-이산화탄소 가스 교환 기관입니다. 세균·바이러스·결핵균의 주요 표적이며, 천식은 기도 과민성으로 발생합니다.",
    "liver":"간은 해독·단백질 합성·담즙 생성의 중심 기관입니다. A~E형 간염 바이러스, 알코올, 담즙 정체 모두 간세포를 손상시킵니다.",
    "stomach":"위는 소화의 시작점입니다. 헬리코박터 파일로리균, 과도한 위산, NSAIDs 복용으로 점막 손상이 일어납니다.",
    "intestine":"소장은 영양 흡수, 대장은 수분 흡수와 배설을 담당합니다. 감염성 장염과 장티푸스는 장 점막을 침범합니다.",
    "kidney":"신장은 혈액 정화, 혈압 조절, 전해질 균형을 담당합니다. 당뇨·고혈압의 장기 합병증으로 신기능 저하가 발생합니다.",
    "skin":"피부는 인체 최대 기관으로 물리적 방어막 역할을 합니다. 진균·세균·바이러스 감염, 자가면역 질환이 피부에 나타납니다.",
    "joints":"관절은 뼈와 뼈 사이 연결부입니다. 연골 마모(골관절염)와 자가면역 염증(류마티스)으로 통증·강직이 발생합니다.",
    "thyroid":"갑상선은 목 앞 나비 모양 호르몬 분비 기관입니다. 호르몬 과다(항진)와 부족(저하) 모두 전신 대사에 영향을 줍니다.",
    "pancreas":"췌장은 인슐린·글루카곤 분비로 혈당을 조절합니다. 인슐린 부족 또는 저항성 증가가 당뇨의 핵심 기전입니다.",
    "lymph":"림프계는 전신 면역 네트워크입니다. 감염 시 림프절 부종, AIDS에서는 CD4 T세포가 직접 파괴됩니다.",
    "blood":"혈액은 산소·영양·면역세포를 운반합니다. 말라리아와 뎅기열은 혈액을 직접 침범하여 적혈구와 혈소판을 손상시킵니다.",
    "spine":"척추는 몸의 중심 기둥입니다. 경추 척추증은 경추 추간판 변성으로 목·어깨·팔로 방사통이 퍼집니다.",
    "eye":"눈은 편두통(눈 뒤 통증, 광선 공포증), 당뇨 합병증(망막병증), 뎅기열(안통)에서 증상이 나타납니다.",
    "nose":"코는 알레르기 비염, 감기의 주요 증상 부위입니다. 비강 점막 염증으로 콧물·코막힘이 발생합니다.",
    "neck":"경추(목 척추)의 추간판 변성이 신경근을 압박하여 목·어깨·팔 통증과 저림이 발생합니다.",
    "esophagus":"식도는 GERD에서 위산 역류로 하부 점막이 손상됩니다. 반복적인 역류는 바렛 식도로 진행할 수 있습니다.",
    "gallbladder":"담낭은 담즙 저장소입니다. 담즙 성분 불균형으로 담석이 형성되고, 담즙 정체 시 황달이 발생합니다.",
    "spleen":"비장은 혈액 필터이자 면역 기관입니다. 말라리아 감염 시 비장 비대(비종대)가 특징적으로 나타납니다.",
    "bladder":"방광은 소변 저장소입니다. 요로 감염은 대부분 대장균이 요도를 통해 방광으로 상행 감염되어 발생합니다.",
    "ear":"내이의 전정기관에 이석(탄산칼슘 결정)이 이탈하면 특정 자세에서 강한 어지럼증이 유발됩니다.",
    "legs":"하지 정맥의 정맥판막 기능 부전으로 혈액이 역류하여 정맥류(혈관 팽창)가 발생합니다.",
    "immune":"면역계는 T세포·B세포·대식세포로 구성됩니다. AIDS에서 HIV가 CD4 T세포를 파괴하여 면역 결핍이 일어납니다.",
}

DISEASE_BODY_PARTS = {
    "Fungal infection":["skin"],
    "Allergy":["skin","lungs","nose"],
    "GERD":["stomach","esophagus"],
    "Chronic cholestasis":["liver","gallbladder"],
    "Drug Reaction":["skin","liver"],
    "Peptic ulcer disease":["stomach"],
    "AIDS":["lymph","blood","immune"],
    "Diabetes":["pancreas","blood","kidney","eye"],
    "Gastroenteritis":["stomach","intestine"],
    "Bronchial Asthma":["lungs"],
    "Hypertension":["heart","blood","kidney","brain"],
    "Migraine":["brain","eye"],
    "Cervical spondylosis":["spine","neck"],
    "Paralysis (brain hemorrhage)":["brain"],
    "Jaundice":["liver","gallbladder","blood"],
    "Malaria":["blood","liver","spleen"],
    "Chicken pox":["skin"],
    "Dengue":["blood","skin","lymph"],
    "Typhoid":["intestine","stomach","blood"],
    "hepatitis A":["liver"],
    "Hepatitis B":["liver"],
    "Hepatitis C":["liver"],
    "Hepatitis D":["liver"],
    "Hepatitis E":["liver"],
    "Alcoholic hepatitis":["liver","stomach"],
    "Tuberculosis":["lungs","lymph"],
    "Common Cold":["nose","lungs"],
    "Pneumonia":["lungs"],
    "Dimorphic hemorrhoids(piles)":["intestine"],
    "Heart attack":["heart"],
    "Varicose veins":["legs","blood"],
    "Hypothyroidism":["thyroid"],
    "Hyperthyroidism":["thyroid"],
    "Hypoglycemia":["pancreas","blood","brain"],
    "Osteoarthritis":["joints","spine"],
    "Arthritis":["joints"],
    "(vertigo) Paroxysmal Positional Vertigo":["ear","brain"],
    "Acne":["skin"],
    "Urinary tract infection":["kidney","bladder"],
    "Psoriasis":["skin","joints"],
    "Impetigo":["skin"],
}

DISEASE_SYMPTOMS = {
    "Fungal infection":["itching","skin_rash","nodal_skin_eruptions","fatigue"],
    "Allergy":["continuous_sneezing","chills","fatigue","cough","redness_of_eyes","sinus_pressure","runny_nose","congestion","headache"],
    "GERD":["stomach_pain","acidity","vomiting","cough","chest_pain","indigestion","headache","nausea"],
    "Chronic cholestasis":["itching","vomiting","fatigue","weight_loss","abdominal_pain","yellowish_skin","dark_urine","nausea"],
    "Drug Reaction":["itching","skin_rash","stomach_pain","vomiting","burning_micturition"],
    "Peptic ulcer disease":["vomiting","indigestion","loss_of_appetite","abdominal_pain","nausea"],
    "AIDS":["muscle_wasting","fatigue","weight_loss","patches_in_throat","sweating","malaise","swelled_lymph_nodes"],
    "Diabetes":["fatigue","weight_loss","restlessness","lethargy","irregular_sugar_level","blurred_and_distorted_vision","weight_gain"],
    "Gastroenteritis":["vomiting","sunken_eyes","dehydration","diarrhoea","nausea"],
    "Bronchial Asthma":["fatigue","cough","breathlessness","phlegm","chest_pain"],
    "Hypertension":["headache","chest_pain","fatigue"],
    "Migraine":["headache","nausea","vomiting","blurred_and_distorted_vision","pain_behind_the_eyes","mood_swings"],
    "Cervical spondylosis":["back_pain","weakness_in_limbs"],
    "Paralysis (brain hemorrhage)":["vomiting","headache","weakness_in_limbs","chest_pain","breathlessness"],
    "Jaundice":["itching","vomiting","fatigue","weight_loss","high_fever","yellowish_skin","dark_urine","abdominal_pain","yellowing_of_eyes"],
    "Malaria":["chills","vomiting","high_fever","sweating","headache","nausea","diarrhoea"],
    "Chicken pox":["itching","skin_rash","fatigue","lethargy","high_fever","headache","loss_of_appetite","mild_fever","swelled_lymph_nodes","malaise"],
    "Dengue":["skin_rash","chills","joint_pain","vomiting","fatigue","high_fever","headache","nausea","loss_of_appetite","pain_behind_the_eyes","back_pain","malaise"],
    "Typhoid":["chills","vomiting","fatigue","high_fever","headache","nausea","constipation","abdominal_pain","diarrhoea"],
    "hepatitis A":["joint_pain","vomiting","yellowish_skin","dark_urine","nausea","loss_of_appetite","abdominal_pain","diarrhoea","mild_fever","yellowing_of_eyes"],
    "Hepatitis B":["itching","fatigue","lethargy","yellowish_skin","dark_urine","loss_of_appetite","abdominal_pain","malaise","yellowing_of_eyes"],
    "Hepatitis C":["fatigue","yellowish_skin","nausea","loss_of_appetite"],
    "Hepatitis D":["joint_pain","vomiting","fatigue","yellowish_skin","dark_urine","nausea","loss_of_appetite","abdominal_pain","yellowing_of_eyes"],
    "Hepatitis E":["joint_pain","vomiting","fatigue","high_fever","yellowish_skin","dark_urine","nausea","loss_of_appetite","abdominal_pain","yellowing_of_eyes","acute_liver_failure"],
    "Alcoholic hepatitis":["vomiting","yellowish_skin","abdominal_pain","swelling_of_stomach","fluid_overload"],
    "Tuberculosis":["chills","vomiting","fatigue","weight_loss","cough","high_fever","breathlessness","sweating","loss_of_appetite","mild_fever","swelled_lymph_nodes","malaise","phlegm"],
    "Common Cold":["continuous_sneezing","chills","fatigue","cough","headache","runny_nose","congestion","mild_fever","malaise","throat_irritation"],
    "Pneumonia":["chills","fatigue","cough","high_fever","breathlessness","sweating","malaise","phlegm","chest_pain","fast_heart_rate"],
    "Dimorphic hemorrhoids(piles)":["constipation","pain_during_bowel_movements"],
    "Heart attack":["vomiting","breathlessness","sweating","chest_pain","fast_heart_rate"],
    "Varicose veins":["fatigue"],
    "Hypothyroidism":["fatigue","weight_gain","cold_hands_and_feets","mood_swings","lethargy"],
    "Hyperthyroidism":["fatigue","mood_swings","weight_loss","restlessness","sweating","fast_heart_rate"],
    "Hypoglycemia":["fatigue","weight_loss","restlessness","cold_hands_and_feets","sweating","irregular_sugar_level","anxiety","blurred_and_distorted_vision","fast_heart_rate"],
    "Osteoarthritis":["joint_pain","back_pain"],
    "Arthritis":["joint_pain","swelled_lymph_nodes"],
    "(vertigo) Paroxysmal Positional Vertigo":["vomiting","headache","nausea"],
    "Acne":["skin_rash"],
    "Urinary tract infection":["burning_micturition","spotting_urination"],
    "Psoriasis":["skin_rash","joint_pain"],
    "Impetigo":["skin_rash","high_fever"],
}

TREATMENT_DB = {
    "Fungal infection":{"drugs":[{"name":"클로트리마졸 (Clotrimazole)","type":"항진균제","note":"약국 구매"},{"name":"테르비나핀 (Terbinafine)","type":"항진균제","note":"발무좀에 효과적"},{"name":"플루코나졸 (Fluconazole)","type":"항진균제(경구)","note":"처방 필요"}],"treatments":["감염 부위 청결·건조","통기성 좋은 면 소재","수건·양말 공유 금지"],"folk_remedies":["티트리 오일 국소 도포","애플사이다 식초 희석 세척","마늘즙 도포"],"urgency":"경과 관찰","urgency_color":"green"},
    "Allergy":{"drugs":[{"name":"세티리진 (Cetirizine)","type":"항히스타민제","note":"약국 구매"},{"name":"로라타딘 (Loratadine)","type":"항히스타민제","note":"비졸림성"},{"name":"나살 스테로이드 스프레이","type":"국소 스테로이드","note":"처방 필요"}],"treatments":["알레르겐 회피","공기청정기 사용","외출 후 세안·샤워"],"folk_remedies":["꿀 소량 섭취","생강차","쿼세틴 함유 식품"],"urgency":"경과 관찰","urgency_color":"green"},
    "GERD":{"drugs":[{"name":"오메프라졸 (Omeprazole)","type":"PPI","note":"위산 억제"},{"name":"파모티딘 (Famotidine)","type":"H2 차단제","note":"약국 구매"},{"name":"탄산칼슘 제산제","type":"제산제","note":"즉각 완화"}],"treatments":["취침 2시간 전 식사 금지","침대 머리 15cm 높이기","카페인·알코올 제한"],"folk_remedies":["알로에베라 주스","생강차","베이킹소다 물"],"urgency":"경과 관찰","urgency_color":"green"},
    "Peptic ulcer disease":{"drugs":[{"name":"오메프라졸","type":"PPI","note":"궤양 치유"},{"name":"수크랄페이트","type":"위점막 보호제","note":"처방 필요"},{"name":"아목시실린+클라리스로마이신","type":"항생제 병합","note":"H.pylori 제거, 처방"}],"treatments":["NSAIDs 중단","금주·금연","H.pylori 검사"],"folk_remedies":["양배추즙","꿀","감초 DGL"],"urgency":"빠른 진료","urgency_color":"orange"},
    "Diabetes":{"drugs":[{"name":"메트포르민 (Metformin)","type":"혈당강하제 1차","note":"처방 필요"},{"name":"인슐린","type":"호르몬 주사제","note":"처방 필요"},{"name":"다파글리플로진","type":"SGLT2억제제","note":"처방 필요"}],"treatments":["혈당 자가 모니터링","저당·저GI 식이","규칙적 유산소 운동","정기 HbA1c 검사"],"folk_remedies":["여주(비터멜론)","계피","차전자피 식전 섭취"],"urgency":"빠른 진료","urgency_color":"orange"},
    "Gastroenteritis":{"drugs":[{"name":"경구수액제 (ORS)","type":"수분 보충","note":"가장 중요"},{"name":"로페라미드","type":"지사제","note":"약국 구매"},{"name":"프로바이오틱스","type":"장내균총 회복","note":"약국 구매"}],"treatments":["충분한 수분·전해질 보충","BRAT 식이","자극적 음식 회피"],"folk_remedies":["생강차","매실청","흰쌀 죽"],"urgency":"경과 관찰","urgency_color":"green"},
    "Bronchial Asthma":{"drugs":[{"name":"살부타몰 흡입제","type":"속효성 기관지확장제","note":"처방 필요"},{"name":"플루티카손 흡입제","type":"흡입 스테로이드","note":"처방 필요"},{"name":"몬테루카스트","type":"류코트리엔 길항제","note":"처방 필요"}],"treatments":["알레르겐 회피","흡입기 올바른 사용","독감 예방접종 매년"],"folk_remedies":["생강차","강황 우유","꿀+검은씨"],"urgency":"빠른 진료","urgency_color":"orange"},
    "Hypertension":{"drugs":[{"name":"암로디핀 (Amlodipine)","type":"칼슘채널차단제","note":"처방 필요"},{"name":"로사르탄 (Losartan)","type":"ARB계","note":"처방 필요"},{"name":"메토프로롤","type":"베타차단제","note":"처방 필요"}],"treatments":["저염식 (하루 5g 미만)","DASH 식이요법","규칙적 유산소 운동","혈압 매일 기록"],"folk_remedies":["마늘 섭취","비트 주스","히비스커스 차"],"urgency":"빠른 진료","urgency_color":"orange"},
    "Migraine":{"drugs":[{"name":"이부프로펜","type":"NSAIDs","note":"약국 구매"},{"name":"수마트립탄","type":"트립탄계","note":"처방 필요"},{"name":"메토클로프라미드","type":"구토억제제","note":"처방 필요"}],"treatments":["어둡고 조용한 환경 휴식","편두통 유발 음식 회피","두통 일지 작성"],"folk_remedies":["마그네슘 보충제","리보플라빈(B2) 고용량","페버퓨 허브"],"urgency":"경과 관찰","urgency_color":"green"},
    "Cervical spondylosis":{"drugs":[{"name":"이부프로펜","type":"NSAIDs","note":"약국 구매"},{"name":"근이완제 (에페리손)","type":"근육 이완제","note":"처방 필요"},{"name":"가바펜틴","type":"신경병증 통증제","note":"처방 필요"}],"treatments":["물리치료·경추 운동","자세 교정","경추 베개 사용","온열 치료"],"folk_remedies":["생강·강황 섭취","캡사이신 크림","온찜질"],"urgency":"경과 관찰","urgency_color":"green"},
    "Paralysis (brain hemorrhage)":{"drugs":[{"name":"119 즉시 호출","type":"응급","note":"골든 타임이 예후 결정"}],"treatments":["119 즉시 호출","FAST 확인","환자 안정 유지","재활 치료 조기 시작"],"folk_remedies":["민간요법 시도 금지"],"urgency":"즉시 병원","urgency_color":"red"},
    "Jaundice":{"drugs":[{"name":"원인 치료 약물 (처방)","type":"원인에 따라 상이","note":"원인 질환 치료가 핵심"},{"name":"우르소데옥시콜산","type":"담즙산","note":"처방 필요"}],"treatments":["반드시 의사 진료","알코올 완전 금지","고지방 음식 회피"],"folk_remedies":["민들레 차","강황 차","비트 주스"],"urgency":"빠른 진료","urgency_color":"orange"},
    "Malaria":{"drugs":[{"name":"아르테미시닌 병합요법 (ACT)","type":"항말라리아제","note":"처방 필요"},{"name":"클로로퀸","type":"항말라리아제","note":"처방 필요"}],"treatments":["즉시 병원 방문","모기 기피제·모기장 사용","수분 보충"],"folk_remedies":["아르테미시아 쑥 차","키나 나무 껍질"],"urgency":"즉시 병원","urgency_color":"red"},
    "Chicken pox":{"drugs":[{"name":"아시클로버 (Acyclovir)","type":"항바이러스제","note":"처방 필요"},{"name":"칼라민 로션","type":"국소 진양제","note":"약국 구매"},{"name":"아세트아미노펜","type":"해열제","note":"아스피린 금지"}],"treatments":["손톱 짧게 유지","헐렁한 면 소재 착용","격리"],"folk_remedies":["오트밀 목욕","베이킹소다 목욕","알로에베라 젤"],"urgency":"경과 관찰","urgency_color":"green"},
    "Dengue":{"drugs":[{"name":"아세트아미노펜","type":"해열진통제","note":"이부프로펜·아스피린 금지"},{"name":"경구수액제 (ORS)","type":"수분 보충","note":"탈수 예방"}],"treatments":["즉시 병원 (혈소판 모니터링)","NSAIDs 절대 금지","충분한 휴식·수분"],"folk_remedies":["파파야 잎 추출물","코코넛 워터"],"urgency":"즉시 병원","urgency_color":"red"},
    "Typhoid":{"drugs":[{"name":"시프로플록사신","type":"항생제","note":"처방 필요"},{"name":"아지트로마이신","type":"항생제","note":"처방 필요"}],"treatments":["즉시 병원 방문","충분한 수분","위생적 음식·물"],"folk_remedies":["바나나","쌀죽","꿀 희석액"],"urgency":"즉시 병원","urgency_color":"red"},
    "hepatitis A":{"drugs":[{"name":"지지 요법 (원인 치료제 없음)","type":"대증 치료","note":"충분한 휴식·수분"},{"name":"A형 간염 백신","type":"예방 백신","note":"노출 후 2주 내"}],"treatments":["충분한 휴식","고단백 저지방 식이","알코올 완전 금지"],"folk_remedies":["민들레 차","밀크씨슬","강황 차"],"urgency":"빠른 진료","urgency_color":"orange"},
    "Hepatitis B":{"drugs":[{"name":"엔테카비르","type":"항바이러스제","note":"처방 필요"},{"name":"테노포비르","type":"항바이러스제","note":"처방 필요"}],"treatments":["간 전문의 방문","알코올 완전 금지","정기 간기능 검사"],"folk_remedies":["밀크씨슬","강황","리코리스 뿌리 차"],"urgency":"빠른 진료","urgency_color":"orange"},
    "Hepatitis C":{"drugs":[{"name":"DAA 병합요법 (소발디+하보니 등)","type":"직접작용 항바이러스제","note":"완치율 95%+, 처방 필요"}],"treatments":["간 전문의 방문 (완치 가능)","알코올 완전 금지"],"folk_remedies":["밀크씨슬","강황"],"urgency":"빠른 진료","urgency_color":"orange"},
    "Alcoholic hepatitis":{"drugs":[{"name":"프레드니솔론","type":"스테로이드","note":"중증, 처방 필요"},{"name":"비타민 B군","type":"영양 보충","note":"약국 구매"}],"treatments":["완전 금주 (가장 중요)","영양 보충","간 전문의 방문"],"folk_remedies":["밀크씨슬","민들레 차"],"urgency":"즉시 병원","urgency_color":"red"},
    "Tuberculosis":{"drugs":[{"name":"HRZE 병합요법","type":"항결핵제 표준","note":"6개월 이상, 처방 필요"}],"treatments":["즉시 병원 (법정 전염병)","격리 치료","완전한 투약 순응"],"folk_remedies":["마늘","강황","홀리 바질"],"urgency":"즉시 병원","urgency_color":"red"},
    "Common Cold":{"drugs":[{"name":"아세트아미노펜","type":"해열진통제","note":"약국 구매"},{"name":"슈도에페드린","type":"충혈완화제","note":"약국 구매"},{"name":"식염수 비강 스프레이","type":"비강 세척","note":"약국 구매"}],"treatments":["충분한 수분 섭취","충분한 휴식","가습기 사용"],"folk_remedies":["꿀+생강+레몬 차","닭고기 수프","아연 로젠지","증기 흡입"],"urgency":"경과 관찰","urgency_color":"green"},
    "Pneumonia":{"drugs":[{"name":"아목시실린","type":"항생제","note":"처방 필요"},{"name":"아지트로마이신","type":"항생제","note":"처방 필요"},{"name":"기관지 확장제","type":"흡입제","note":"처방 필요"}],"treatments":["즉시 병원 방문","충분한 수분 보충","폐렴구균 백신 권장"],"folk_remedies":["따뜻한 증기 흡입","꿀+생강차","프로바이오틱스"],"urgency":"즉시 병원","urgency_color":"red"},
    "Dimorphic hemorrhoids(piles)":{"drugs":[{"name":"하이드로코르티손 좌약","type":"국소 스테로이드","note":"약국 구매"},{"name":"리도카인 연고","type":"국소 마취제","note":"약국 구매"}],"treatments":["고섬유 식이","충분한 수분","온수 좌욕 하루 2~3회"],"folk_remedies":["알로에베라 젤","위치하젤 패드","감자 냉찜질"],"urgency":"경과 관찰","urgency_color":"green"},
    "Heart attack":{"drugs":[{"name":"아스피린 300mg","type":"혈소판 억제제","note":"의심 즉시 씹어서 복용"},{"name":"니트로글리세린 설하정","type":"혈관확장제","note":"처방 있을 때만"}],"treatments":["119 즉시 호출","누운 자세 유지","CPR 준비"],"folk_remedies":["민간요법 시도 금지 — 즉각 119 신고"],"urgency":"즉시 병원","urgency_color":"red"},
    "Varicose veins":{"drugs":[{"name":"디오스민+헤스페리딘","type":"정맥 강화제","note":"약국 구매"},{"name":"경화요법 주사","type":"시술","note":"혈관외과"}],"treatments":["의료용 압박 스타킹","다리 올리기 자세","규칙적 걷기 운동"],"folk_remedies":["말밤나무 추출물","포도씨 추출물","사과식초 국소 도포"],"urgency":"경과 관찰","urgency_color":"green"},
    "Hypothyroidism":{"drugs":[{"name":"레보티록신 (Levothyroxine)","type":"갑상선 호르몬 보충제","note":"처방 필요, 공복 복용"}],"treatments":["정기 TSH 검사","공복 복용","규칙적 운동"],"folk_remedies":["셀레늄 보충","아슈와간다 허브","김·미역 (요오드)"],"urgency":"빠른 진료","urgency_color":"orange"},
    "Hyperthyroidism":{"drugs":[{"name":"메티마졸","type":"항갑상선제","note":"처방 필요"},{"name":"프로프라놀롤","type":"베타차단제","note":"처방 필요"}],"treatments":["정기 갑상선 기능 검사","요오드 함유 식품 제한","카페인 제한"],"folk_remedies":["레몬밤 차","버그위드 허브","브로콜리·배추"],"urgency":"빠른 진료","urgency_color":"orange"},
    "Hypoglycemia":{"drugs":[{"name":"포도당 15~20g 즉각 섭취","type":"응급 처치","note":"주스·사탕·설탕물"},{"name":"글루카곤 키트","type":"응급 주사제","note":"처방 필요"}],"treatments":["15-15 규칙","규칙적 식사","혈당측정기 휴대"],"folk_remedies":["꿀 1~2 티스푼 즉각","바나나","오트밀"],"urgency":"빠른 진료","urgency_color":"orange"},
    "Osteoarthritis":{"drugs":[{"name":"아세트아미노펜","type":"진통제 1차","note":"약국 구매"},{"name":"이부프로펜","type":"NSAIDs","note":"약국 구매"},{"name":"글루코사민+콘드로이친","type":"관절 보호제","note":"약국 구매"}],"treatments":["적정 체중 유지","저충격 운동 (수영·자전거)","온열·냉찜질"],"folk_remedies":["생강·강황 섭취","유황 온천","아보카도-소야 추출물"],"urgency":"경과 관찰","urgency_color":"green"},
    "Arthritis":{"drugs":[{"name":"이부프로펜","type":"NSAIDs","note":"약국 구매"},{"name":"메토트렉세이트","type":"DMARD","note":"처방 필요"}],"treatments":["적정 체중 유지","규칙적 관절 운동","물리치료"],"folk_remedies":["강황 + 흑후추","생강차","오메가-3 어유"],"urgency":"빠른 진료","urgency_color":"orange"},
    "(vertigo) Paroxysmal Positional Vertigo":{"drugs":[{"name":"메클리진","type":"항히스타민제","note":"처방 필요"},{"name":"디멘히드리네이트","type":"어지럼 완화","note":"약국 구매"}],"treatments":["엡리 이석 정복술 (Epley maneuver)","갑작스런 머리 움직임 피하기","이비인후과·신경과 방문"],"folk_remedies":["생강차","은행 추출물","충분한 수분"],"urgency":"빠른 진료","urgency_color":"orange"},
    "Acne":{"drugs":[{"name":"벤조일퍼옥사이드","type":"국소 항균제","note":"약국 구매"},{"name":"트레티노인","type":"국소 레티노이드","note":"처방 필요"},{"name":"독시사이클린","type":"경구 항생제","note":"처방 필요"}],"treatments":["하루 2회 순한 클렌저 세안","손으로 짜지 않기","비코메도제닉 제품 사용"],"folk_remedies":["티트리 오일","알로에베라 젤","녹차 추출물 토너"],"urgency":"경과 관찰","urgency_color":"green"},
    "Urinary tract infection":{"drugs":[{"name":"트리메토프림-설파메톡사졸","type":"항생제","note":"처방 필요"},{"name":"니트로푸란토인","type":"항생제","note":"처방 필요"},{"name":"페나조피리딘","type":"진통제 (요도)","note":"배뇨 통증 완화"}],"treatments":["충분한 수분 (하루 2L+)","앞→뒤 방향 회음부","카페인·알코올 제한"],"folk_remedies":["크랜베리 주스","D-만노스 보충제","프로바이오틱스"],"urgency":"빠른 진료","urgency_color":"orange"},
    "Psoriasis":{"drugs":[{"name":"코르티코스테로이드 크림","type":"국소 스테로이드","note":"처방 필요"},{"name":"칼시포트리올","type":"비타민D 유도체","note":"처방 필요"}],"treatments":["순한 보습제 매일 사용","스트레스 관리","자외선 치료"],"folk_remedies":["알로에베라 젤","어성초 크림","오트밀 목욕","오메가-3 보충"],"urgency":"경과 관찰","urgency_color":"green"},
    "Impetigo":{"drugs":[{"name":"무피로신 연고","type":"국소 항생제","note":"처방 필요"},{"name":"세팔렉신","type":"경구 항생제","note":"처방 필요"}],"treatments":["병변 청결 유지","수건·침구 개인 사용","등원 금지 (완치 전)"],"folk_remedies":["꿀 국소 도포","알로에베라 젤","강황 페이스트"],"urgency":"빠른 진료","urgency_color":"orange"},
    "Chronic cholestasis":{"drugs":[{"name":"우르소데옥시콜산 (UDCA)","type":"담즙산","note":"처방 필요"},{"name":"콜레스티라민","type":"담즙산 결합제","note":"처방 필요"}],"treatments":["간담도 전문의 방문","지용성 비타민 보충","저지방 식이"],"folk_remedies":["민들레 차","아티초크 차","강황"],"urgency":"빠른 진료","urgency_color":"orange"},
    "Drug Reaction":{"drugs":[{"name":"원인 약물 즉시 중단","type":"1차 처치","note":"의사 상담"},{"name":"항히스타민제","type":"증상 완화","note":"경증 피부 반응"},{"name":"에피네프린 자동주사기","type":"응급","note":"아나필락시스 시"}],"treatments":["원인 약물 식별·기록","아나필락시스 시 119 즉시"],"folk_remedies":["냉찜질","알로에베라 젤"],"urgency":"빠른 진료","urgency_color":"orange"},
    "AIDS":{"drugs":[{"name":"항레트로바이러스 요법 (ART)","type":"복합 항바이러스 요법","note":"즉시 시작, 처방 필요"}],"treatments":["즉시 감염내과 방문","ART 복약 순응","기회감염 예방"],"folk_remedies":["강황 (보조)","충분한 수면","균형 잡힌 영양"],"urgency":"즉시 병원","urgency_color":"red"},
    "Hepatitis D":{"drugs":[{"name":"페그인터페론 알파","type":"면역조절제","note":"처방 필요"}],"treatments":["간 전문의 방문","알코올 완전 금지"],"folk_remedies":["밀크씨슬","강황"],"urgency":"빠른 진료","urgency_color":"orange"},
    "Hepatitis E":{"drugs":[{"name":"지지 요법","type":"대증 치료","note":"면역 정상인 자연 회복"},{"name":"리바비린","type":"항바이러스제","note":"처방 필요"}],"treatments":["충분한 휴식","알코올 금지","임산부 즉시 병원"],"folk_remedies":["민들레 차","강황"],"urgency":"빠른 진료","urgency_color":"orange"},
}

# ════════════════════════════════════════════════════════
#  ML
# ════════════════════════════════════════════════════════
@st.cache_data
def build_training_data():
    all_syms = sorted(set(s for v in DISEASE_SYMPTOMS.values() for s in v))
    rows, rng = [], np.random.default_rng(42)
    for disease, syms in DISEASE_SYMPTOMS.items():
        for _ in range(30):
            row = {s: 0 for s in all_syms}
            n = max(2, int(len(syms)*rng.uniform(0.65,1.0)))
            for s in rng.choice(syms, size=min(n,len(syms)), replace=False): row[s]=1
            pool = [s for s in all_syms if s not in syms]
            if pool:
                for ns in rng.choice(pool, size=rng.integers(0,3), replace=False): row[ns]=1
            row["disease"] = disease
            rows.append(row)
    return pd.DataFrame(rows), all_syms

@st.cache_resource
def train_models():
    df, all_syms = build_training_data()
    X, y = df[all_syms].values, df["disease"].values
    nb = GaussianNB(); nb.fit(X, y)
    rf = RandomForestClassifier(n_estimators=120, random_state=42); rf.fit(X, y)
    return nb, rf, all_syms

# ════════════════════════════════════════════════════════
#  인체 SVG 해부도 렌더러

# ════════════════════════════════════════════════════════
#  고퀄리티 인체 해부도 렌더러 v2
# ════════════════════════════════════════════════════════
BODY_PART_FULL_INFO = {
    "brain":      {"name":"뇌","sub":"중추신경계 · 두개골 내부","desc":"뇌는 1,000억 개 이상의 신경세포로 이루어진 인체 최고의 제어 센터입니다. 편두통은 뇌혈관의 수축·확장으로 발생하고, 뇌출혈은 혈관 파열로 뇌 조직이 손상되는 응급 상황입니다."},
    "heart":      {"name":"심장","sub":"흉강 좌측 · 근육 펌프","desc":"성인 심장은 하루 약 10만 번 박동하며 5L의 혈액을 순환시킵니다. 심근경색은 관상동맥이 막혀 심근 조직이 괴사하는 응급 상황으로 골든 타임은 90분입니다."},
    "lungs":      {"name":"폐","sub":"흉강 양측 · 가스 교환","desc":"좌폐 2엽, 우폐 3엽으로 구성되며 폐포 면적은 테니스코트 크기(70㎡)에 달합니다. 폐렴은 폐포를 세균·바이러스가 침범하고, 결핵균은 폐 상엽을 주로 침범합니다."},
    "liver":      {"name":"간","sub":"복강 우상부 · 해독·대사","desc":"체내 최대 장기(1.2~1.5kg)로 500가지 이상의 대사 기능을 담당합니다. A~E형 간염 바이러스 모두 간세포를 표적으로 삼으며, 만성 염증 반복 시 간경화로 진행됩니다."},
    "stomach":    {"name":"위","sub":"복강 좌상부 · 소화 시작","desc":"위는 강산성(pH 1~3) 위액으로 음식을 분해합니다. 헬리코박터 파일로리균이 위 점막 보호층을 무너뜨려 궤양을 유발하고, GERD는 하부식도괄약근 기능 이상으로 발생합니다."},
    "intestine":  {"name":"소장·대장","sub":"복강 중앙 · 흡수·배설","desc":"소장은 6~7m로 영양소를 흡수하고, 대장은 수분 재흡수·배설을 담당합니다. 장티푸스균은 소장 림프절에 침입하고, 치질은 항문 주위 정맥총이 팽창해 발생합니다."},
    "kidney":     {"name":"신장","sub":"후복막 · 혈액 정화","desc":"양쪽 신장은 하루 180L의 혈액을 여과해 1~2L의 소변을 생성합니다. 당뇨병성 신증과 고혈압성 신경화증은 만성 신부전의 주요 원인입니다."},
    "skin":       {"name":"피부","sub":"전신 표면 · 1차 방어막","desc":"체표면적 약 1.5~2㎡, 무게 약 3~4kg으로 인체 최대 기관입니다. 건선은 T세포 과활성화로 피부세포가 과증식하고, 여드름은 피지선 과분비와 세균 감염으로 발생합니다."},
    "joints":     {"name":"관절","sub":"뼈 연결부 · 운동 담당","desc":"초자연골·활액막·인대·건으로 구성됩니다. 골관절염은 연골 마모에 의한 퇴행성 질환이고, 류마티스 관절염은 활액막을 자가면역 반응이 공격하는 전신 염증 질환입니다."},
    "thyroid":    {"name":"갑상선","sub":"경부 전면 · 호르몬 분비","desc":"나비 모양의 내분비 기관으로 T3·T4 호르몬을 분비합니다. 기능 항진은 대사항진(체중 감소·빠른 심박), 기능 저하는 대사저하(피로·체중 증가) 증상으로 나타납니다."},
    "pancreas":   {"name":"췌장","sub":"복강 후방 · 혈당 조절","desc":"랑게르한스섬의 β세포가 인슐린을 분비합니다. 1형 당뇨는 β세포 자가면역 파괴, 2형 당뇨는 인슐린 저항성 증가가 핵심 기전입니다."},
    "gallbladder":{"name":"담낭","sub":"간 하면 · 담즙 저장","desc":"담즙을 농축·저장했다가 지방 소화 시 십이지장으로 분비합니다. 담즙 성분 불균형이나 정체 시 담석이 형성되고, 담즙이 혈중으로 역류하면 황달이 발생합니다."},
    "spleen":     {"name":"비장","sub":"복강 좌상부 · 면역 필터","desc":"노화된 적혈구를 제거하고 면역세포를 생산합니다. 말라리아 감염 시 감염된 적혈구가 비장에 집적되어 비종대(비장 비대)가 특징적으로 나타납니다."},
    "bladder":    {"name":"방광","sub":"골반강 · 소변 저장","desc":"400~600ml의 소변을 저장합니다. 요로 감염의 80%는 대장균이 요도를 통해 상행 감염되며, 여성은 요도가 짧아 감염 위험이 높습니다."},
    "spine":      {"name":"척추","sub":"중심축 · 신경 보호","desc":"경추 7개, 흉추 12개, 요추 5개의 척추뼈로 구성됩니다. 경추 척추증은 추간판 변성으로 신경근이 압박되어 목·어깨·팔로 방사통이 발생합니다."},
    "legs":       {"name":"하지·정맥","sub":"하체 · 혈액 환류","desc":"하지 정맥판막이 일방통행 밸브 역할을 합니다. 정맥류는 판막 기능 부전으로 혈액이 역류하면서 정맥벽이 팽창·구불구불해지는 질환입니다."},
    "eye":        {"name":"눈","sub":"시각기관 · 안구","desc":"편두통에서 광선 공포증·눈 앞 섬광 전조가 나타납니다. 당뇨병성 망막병증은 망막 미세혈관이 손상되어 진행하는 성인 실명의 주요 원인입니다."},
    "ear":        {"name":"귀 (내이)","sub":"청각·전정기관","desc":"내이 전정기관의 이석(탄산칼슘 결정)이 반고리관으로 이탈하면 특정 자세에서 강한 회전성 어지럼증이 유발됩니다."},
    "esophagus":  {"name":"식도","sub":"인두~위 연결 통로","desc":"25cm 길이의 근육관으로 하부식도괄약근이 위산 역류를 막습니다. GERD에서 반복적인 위산 자극은 바렛 식도, 나아가 식도암으로 진행할 수 있습니다."},
    "blood":      {"name":"혈액","sub":"전신 순환 · 운반 매체","desc":"적혈구(산소)·백혈구(면역)·혈소판(지혈)·혈장으로 구성됩니다. 말라리아 원충은 적혈구 내 증식·파열하고, 뎅기열은 혈소판을 감소시켜 출혈 경향을 높입니다."},
    "lymph":      {"name":"림프계","sub":"면역 네트워크","desc":"림프관·림프절·비장·흉선으로 구성된 면역 고속도로입니다. 감염 시 림프절에서 T·B세포가 활성화되며, AIDS에서는 HIV가 CD4 T세포를 직접 파괴합니다."},
    "immune":     {"name":"면역계","sub":"전신 방어 체계","desc":"선천면역(대식세포·NK세포)과 후천면역(T세포·B세포)으로 구성됩니다. HIV는 CD4 T세포에 침투해 점진적으로 파괴하고, CD4 수치 200/μL 미만이면 AIDS로 진행됩니다."},
    "neck":       {"name":"경추","sub":"목 척추","desc":"경추 7개 척추뼈가 뇌와 몸통을 연결합니다. 추간판 탈수·변성으로 신경근이 압박되면 목·어깨·팔 방사통이 발생합니다."},
    "nose":       {"name":"코","sub":"호흡·후각기관","desc":"비강 점막은 공기를 가온·가습·여과합니다. 알레르기 비염은 알레르겐에 반응한 비만세포가 히스타민을 분비해 콧물·재채기·코막힘을 유발합니다."},
}


def render_body_anatomy(active_parts: dict, part_disease_map: dict):
    """고퀄리티 인터랙티브 인체 해부도 (v2)"""

    # JS용 데이터 직렬화
    part_info_js = {}
    for part, info in BODY_PART_FULL_INFO.items():
        diseases = sorted(part_disease_map.get(part, []), key=lambda x: -x["prob"])[:6]
        part_info_js[part] = {
            "name": info["name"],
            "sub":  info["sub"],
            "desc": info["desc"],
            "intensity": round(active_parts.get(part, 0) * 100),
            "diseases": diseases,
        }

    active_json   = json.dumps({p: round(v, 3) for p, v in active_parts.items()})
    part_info_json = json.dumps(part_info_js, ensure_ascii=False)

    html = """<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,sans-serif;background:transparent;}
.wrap{display:flex;gap:20px;align-items:flex-start;}
.svg-col{flex:0 0 270px;}
.panel-col{flex:1;min-width:0;}
.info-card{background:#fff;border:0.5px solid rgba(0,0,0,.1);border-radius:14px;padding:18px;margin-bottom:10px;}
.info-name{font-size:20px;font-weight:500;color:#111;margin-bottom:2px;}
.info-sub{font-size:12px;color:#888;margin-bottom:14px;}
.int-wrap{display:flex;align-items:center;gap:10px;margin-bottom:14px;}
.int-label{font-size:11px;color:#888;font-weight:500;text-transform:uppercase;letter-spacing:.06em;min-width:44px;}
.int-track{flex:1;height:7px;background:#eee;border-radius:4px;overflow:hidden;}
.int-fill{height:100%;border-radius:4px;transition:width .45s cubic-bezier(.4,0,.2,1),background .3s;}
.int-pct{font-size:12px;font-weight:500;min-width:34px;text-align:right;}
.sec-label{font-size:10px;font-weight:600;color:#999;text-transform:uppercase;letter-spacing:.08em;margin-bottom:7px;}
.desc{font-size:13px;color:#555;line-height:1.68;margin-bottom:14px;}
.tag-wrap{display:flex;flex-wrap:wrap;gap:5px;}
.tag{font-size:11px;padding:3px 11px;border-radius:999px;font-weight:500;border:1px solid;}
.tag-r{background:#FCEBEB;color:#791F1F;border-color:#F09595;}
.tag-o{background:#FAEEDA;color:#633806;border-color:#EF9F27;}
.tag-g{background:#EAF3DE;color:#27500A;border-color:#97C459;}
.tag-n{background:#f5f5f3;color:#888;border-color:rgba(0,0,0,.1);}
.legend{display:flex;gap:14px;margin-top:12px;font-size:11px;color:#888;flex-wrap:wrap;}
.lbullet{width:12px;height:12px;border-radius:50%;display:inline-block;margin-right:3px;vertical-align:middle;}
.hint{font-size:11px;color:#aaa;text-align:center;margin-top:6px;}
.oz{cursor:pointer;transition:opacity .16s;}
.oz:hover{opacity:.7;}
@media(prefers-color-scheme:dark){
  .info-card{background:#1e1e1c;border-color:rgba(255,255,255,.1);}
  .info-name{color:#eee;} .info-sub,.desc,.int-label,.sec-label,.hint{color:#aaa;}
  .int-pct{color:#ccc;} .int-track{background:#333;}
  .tag-r{background:#501313;color:#F7C1C1;border-color:#A32D2D;}
  .tag-o{background:#412402;color:#FAC775;border-color:#854F0B;}
  .tag-g{background:#173404;color:#C0DD97;border-color:#3B6D11;}
  .tag-n{background:#2a2a28;color:#aaa;border-color:rgba(255,255,255,.1);}
  .legend{color:#999;}
}
</style></head><body>
<div class="wrap">
<div class="svg-col">
<svg viewBox="0 0 270 740" width="100%" style="display:block;">
<defs>
  <linearGradient id="sg" x1="0" y1="0" x2=".3" y2="1">
    <stop offset="0%" stop-color="#F5D0A8"/>
    <stop offset="100%" stop-color="#E0A070"/>
  </linearGradient>
  <linearGradient id="sg2" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%" stop-color="#EEC090"/>
    <stop offset="100%" stop-color="#D49060"/>
  </linearGradient>
</defs>

<!-- 몸 실루엣 -->
<path d="M95,138 Q80,134 70,143 L52,165 L40,215 L38,275 L42,335 L51,365 L60,374 L68,363 L70,300 L74,240 L80,182 L80,162 L102,156" fill="url(#sg)" stroke="#C88A60" stroke-width="1.2"/>
<path d="M175,138 Q190,134 200,143 L218,165 L230,215 L232,275 L228,335 L219,365 L210,374 L202,363 L200,300 L196,240 L190,182 L190,162 L168,156" fill="url(#sg)" stroke="#C88A60" stroke-width="1.2"/>
<path d="M80,160 L102,154 L135,152 L168,154 L190,160 L194,185 L196,225 L196,318 L193,375 L188,394 L178,396 L178,428 L188,500 L196,570 L200,640 L204,708 L66,708 L70,640 L74,570 L82,500 L92,428 L92,396 L82,394 L77,375 L74,318 L74,225 L76,185 Z" fill="url(#sg)" stroke="#C88A60" stroke-width="1.3"/>
<rect x="114" y="111" width="42" height="47" rx="12" fill="url(#sg)" stroke="#C88A60" stroke-width="1.2"/>
<ellipse cx="135" cy="69" rx="52" ry="62" fill="url(#sg)" stroke="#C88A60" stroke-width="1.3"/>
<ellipse cx="83" cy="69" rx="9" ry="14" fill="#E8B888" stroke="#C88A60" stroke-width="1"/>
<ellipse cx="187" cy="69" rx="9" ry="14" fill="#E8B888" stroke="#C88A60" stroke-width="1"/>

<!-- 얼굴 -->
<ellipse cx="120" cy="74" rx="9.5" ry="6.5" fill="#fff" stroke="#B89070" stroke-width=".8"/>
<ellipse cx="150" cy="74" rx="9.5" ry="6.5" fill="#fff" stroke="#B89070" stroke-width=".8"/>
<circle cx="120" cy="74" r="4.5" fill="#3A2820"/>
<circle cx="150" cy="74" r="4.5" fill="#3A2820"/>
<circle cx="121" cy="73" r="1.4" fill="white"/>
<circle cx="151" cy="73" r="1.4" fill="white"/>
<path d="M130,87 Q135,93 140,87" fill="none" stroke="#B87860" stroke-width="1.3" stroke-linecap="round"/>
<path d="M126,85 Q128,90 130,87" fill="none" stroke="#C09080" stroke-width="1" stroke-linecap="round"/>
<path d="M140,87 Q142,90 144,85" fill="none" stroke="#C09080" stroke-width="1" stroke-linecap="round"/>
<ellipse cx="129" cy="92" rx="4.5" ry="3.5" fill="#D49878" opacity=".65"/>
<ellipse cx="141" cy="92" rx="4.5" ry="3.5" fill="#D49878" opacity=".65"/>

<!-- 손발 -->
<ellipse cx="47" cy="382" rx="13" ry="17" fill="url(#sg)" stroke="#C88A60" stroke-width="1"/>
<ellipse cx="223" cy="382" rx="13" ry="17" fill="url(#sg)" stroke="#C88A60" stroke-width="1"/>
<path d="M88,708 Q70,708 66,700 L62,692 L64,686 L96,686 L98,708 Z" fill="url(#sg)" stroke="#C88A60" stroke-width="1"/>
<path d="M182,708 Q200,708 204,700 L208,692 L206,686 L174,686 L172,708 Z" fill="url(#sg)" stroke="#C88A60" stroke-width="1"/>

<!-- 근육 결 -->
<path d="M80,205 Q78,240 80,270" fill="none" stroke="#C89070" stroke-width=".5" opacity=".4"/>
<path d="M190,205 Q192,240 190,270" fill="none" stroke="#C89070" stroke-width=".5" opacity=".4"/>
<path d="M100,164 L100,394" fill="none" stroke="#C89070" stroke-width=".35" opacity=".22"/>
<path d="M170,164 L170,394" fill="none" stroke="#C89070" stroke-width=".35" opacity=".22"/>

<!-- ══ 장기 ══ -->

<!-- 뇌 -->
<g class="oz" id="oz-brain" data-part="brain">
  <ellipse cx="135" cy="57" rx="38" ry="34" fill="#C8D8F8" stroke="#5868C8" stroke-width="1.6"/>
  <path d="M100,53 Q112,36 135,34 Q158,36 170,53" fill="none" stroke="#7888D8" stroke-width="1" opacity=".65"/>
  <path d="M102,63 Q115,49 135,47 Q155,49 168,63" fill="none" stroke="#7888D8" stroke-width=".8" opacity=".5"/>
  <path d="M103,72 Q116,61 135,60 Q154,61 167,72" fill="none" stroke="#7888D8" stroke-width=".6" opacity=".38"/>
  <line x1="135" y1="24" x2="135" y2="90" stroke="#8898D8" stroke-width=".7" opacity=".3"/>
  <text x="135" y="62" text-anchor="middle" font-size="10" font-weight="600" fill="#2838A0" font-family="-apple-system,sans-serif">뇌</text>
</g>

<!-- 눈 -->
<g class="oz" id="oz-eye" data-part="eye">
  <ellipse cx="120" cy="74" rx="9.5" ry="6.5" fill="#A0D0F0" stroke="#3888C8" stroke-width="1.1"/>
  <ellipse cx="150" cy="74" rx="9.5" ry="6.5" fill="#A0D0F0" stroke="#3888C8" stroke-width="1.1"/>
</g>

<!-- 귀 -->
<g class="oz" id="oz-ear" data-part="ear">
  <ellipse cx="83" cy="69" rx="9" ry="14" fill="#F0C898" stroke="#B87848" stroke-width="1.1" opacity=".85"/>
  <ellipse cx="187" cy="69" rx="9" ry="14" fill="#F0C898" stroke="#B87848" stroke-width="1.1" opacity=".85"/>
  <path d="M85,62 Q89,69 85,76" fill="none" stroke="#A06838" stroke-width=".9" opacity=".7"/>
  <path d="M185,62 Q181,69 185,76" fill="none" stroke="#A06838" stroke-width=".9" opacity=".7"/>
</g>

<!-- 갑상선 -->
<g class="oz" id="oz-thyroid" data-part="thyroid">
  <path d="M120,132 Q113,125 113,133 Q113,144 124,147 L135,148 L146,147 Q157,144 157,133 Q157,125 150,132 Q146,136 135,137 Q124,136 120,132 Z" fill="#F8C090" stroke="#E09050" stroke-width="1.3"/>
  <text x="135" y="142" text-anchor="middle" font-size="7.5" font-weight="600" fill="#8A3A08" font-family="-apple-system,sans-serif">갑상선</text>
</g>

<!-- 기관 -->
<g class="oz" id="oz-esophagus" data-part="esophagus">
  <rect x="132" y="152" width="6" height="55" rx="3" fill="#D8C888" stroke="#A89848" stroke-width=".9"/>
</g>

<!-- 척추 -->
<g class="oz" id="oz-spine" data-part="spine">
  <rect x="132" y="160" width="6" height="234" rx="2.5" fill="#EDE8D0" stroke="#A0987A" stroke-width=".9"/>
  <rect x="130" y="167" width="10" height="7" rx="2.2" fill="#EDE8D0" stroke="#A0987A" stroke-width=".7"/>
  <rect x="130" y="188" width="10" height="7" rx="2.2" fill="#EDE8D0" stroke="#A0987A" stroke-width=".7"/>
  <rect x="130" y="209" width="10" height="7" rx="2.2" fill="#EDE8D0" stroke="#A0987A" stroke-width=".7"/>
  <rect x="130" y="230" width="10" height="7" rx="2.2" fill="#EDE8D0" stroke="#A0987A" stroke-width=".7"/>
  <rect x="130" y="251" width="10" height="7" rx="2.2" fill="#EDE8D0" stroke="#A0987A" stroke-width=".7"/>
  <rect x="130" y="272" width="10" height="7" rx="2.2" fill="#EDE8D0" stroke="#A0987A" stroke-width=".7"/>
  <rect x="130" y="293" width="10" height="7" rx="2.2" fill="#EDE8D0" stroke="#A0987A" stroke-width=".7"/>
  <rect x="130" y="314" width="10" height="7" rx="2.2" fill="#EDE8D0" stroke="#A0987A" stroke-width=".7"/>
  <rect x="130" y="335" width="10" height="7" rx="2.2" fill="#EDE8D0" stroke="#A0987A" stroke-width=".7"/>
  <rect x="130" y="356" width="10" height="7" rx="2.2" fill="#EDE8D0" stroke="#A0987A" stroke-width=".7"/>
  <rect x="130" y="377" width="10" height="7" rx="2.2" fill="#EDE8D0" stroke="#A0987A" stroke-width=".7"/>
</g>

<!-- 폐 -->
<g class="oz" id="oz-lungs" data-part="lungs">
  <path d="M102,165 Q86,168 80,186 L76,240 Q74,272 87,286 Q100,296 115,292 L119,258 L120,165 Z" fill="#F0A098" stroke="#B84840" stroke-width="1.6"/>
  <path d="M168,165 Q184,168 190,186 L194,240 Q196,272 183,286 Q170,296 155,292 L151,258 L150,165 Z" fill="#F0A098" stroke="#B84840" stroke-width="1.6"/>
  <path d="M89,195 Q93,214 91,246" fill="none" stroke="#B84840" stroke-width=".7" opacity=".5"/>
  <path d="M100,189 Q104,213 102,252" fill="none" stroke="#B84840" stroke-width=".6" opacity=".4"/>
  <path d="M181,195 Q177,214 179,246" fill="none" stroke="#B84840" stroke-width=".7" opacity=".5"/>
  <path d="M170,189 Q166,213 168,252" fill="none" stroke="#B84840" stroke-width=".6" opacity=".4"/>
  <text x="98" y="237" text-anchor="middle" font-size="9" font-weight="600" fill="#801818" font-family="-apple-system,sans-serif">좌폐</text>
  <text x="172" y="237" text-anchor="middle" font-size="9" font-weight="600" fill="#801818" font-family="-apple-system,sans-serif">우폐</text>
</g>

<!-- 심장 -->
<g class="oz" id="oz-heart" data-part="heart">
  <path d="M135,188 Q118,177 106,188 Q93,200 106,216 L135,246 L164,216 Q177,200 164,188 Q152,177 135,188 Z" fill="#E03030" stroke="#981010" stroke-width="2"/>
  <path d="M118,196 Q112,206 118,218 L128,228" fill="none" stroke="#FF7070" stroke-width="1" opacity=".4"/>
  <!-- 대동맥 -->
  <path d="M135,179 Q136,167 140,160 Q144,150 142,145" fill="none" stroke="#C01010" stroke-width="2.2" stroke-linecap="round"/>
  <!-- 폐동맥 -->
  <path d="M128,186 L110,171" fill="none" stroke="#4868C8" stroke-width="1.6" stroke-linecap="round"/>
  <path d="M142,186 L160,171" fill="none" stroke="#C83838" stroke-width="1.6" stroke-linecap="round"/>
  <text x="135" y="214" text-anchor="middle" font-size="9" font-weight="600" fill="#7A0808" font-family="-apple-system,sans-serif">심장</text>
</g>

<!-- 간 -->
<g class="oz" id="oz-liver" data-part="liver">
  <path d="M155,256 Q177,249 192,258 L196,287 Q192,314 174,316 Q158,316 148,306 Q140,295 146,260 Z" fill="#C06838" stroke="#8A3810" stroke-width="1.6"/>
  <path d="M163,265 Q176,262 186,268" fill="none" stroke="#E09050" stroke-width=".8" opacity=".55"/>
  <path d="M161,278 Q174,275 186,281" fill="none" stroke="#E09050" stroke-width=".7" opacity=".45"/>
  <text x="170" y="290" text-anchor="middle" font-size="9" font-weight="600" fill="#5A2808" font-family="-apple-system,sans-serif">간</text>
</g>

<!-- 담낭 -->
<g class="oz" id="oz-gallbladder" data-part="gallbladder">
  <ellipse cx="180" cy="322" rx="11" ry="14" fill="#A8C858" stroke="#608018" stroke-width="1.3"/>
  <path d="M175,314 Q180,311 185,314" fill="none" stroke="#608018" stroke-width=".8" opacity=".7"/>
  <text x="180" y="325" text-anchor="middle" font-size="7" font-weight="600" fill="#386008" font-family="-apple-system,sans-serif">담낭</text>
</g>

<!-- 위 -->
<g class="oz" id="oz-stomach" data-part="stomach">
  <path d="M112,261 Q95,264 89,282 Q83,304 97,318 Q110,328 133,326 L138,293 L128,261 Z" fill="#E89848" stroke="#A85818" stroke-width="1.6"/>
  <path d="M98,280 Q96,294 100,308" fill="none" stroke="#C87828" stroke-width=".8" opacity=".55"/>
  <text x="110" y="298" text-anchor="middle" font-size="9" font-weight="600" fill="#703208" font-family="-apple-system,sans-serif">위</text>
</g>

<!-- 비장 -->
<g class="oz" id="oz-spleen" data-part="spleen">
  <ellipse cx="88" cy="303" rx="15" ry="19" fill="#9870C0" stroke="#5030A0" stroke-width="1.3"/>
  <path d="M82,296 Q88,300 94,296" fill="none" stroke="#7050A8" stroke-width=".8" opacity=".7"/>
  <text x="88" y="307" text-anchor="middle" font-size="7.5" font-weight="600" fill="#380880" font-family="-apple-system,sans-serif">비장</text>
</g>

<!-- 췌장 -->
<g class="oz" id="oz-pancreas" data-part="pancreas">
  <path d="M110,330 Q133,323 163,327 L166,340 Q140,348 114,344 Z" fill="#E8B868" stroke="#B87828" stroke-width="1.3"/>
  <text x="138" y="339" text-anchor="middle" font-size="8" font-weight="600" fill="#784808" font-family="-apple-system,sans-serif">췌장</text>
</g>

<!-- 신장 -->
<g class="oz" id="oz-kidney" data-part="kidney">
  <path d="M90,345 Q77,345 75,358 Q73,375 84,383 Q92,389 101,383 L103,358 Q103,345 90,345 Z" fill="#D05880" stroke="#903060" stroke-width="1.6"/>
  <path d="M84,356 Q82,367 86,376" fill="none" stroke="#F07898" stroke-width=".8" opacity=".55"/>
  <text x="87" y="368" text-anchor="middle" font-size="7.5" font-weight="600" fill="#620030" font-family="-apple-system,sans-serif">신장</text>
  <path d="M180,345 Q193,345 195,358 Q197,375 186,383 Q178,389 169,383 L167,358 Q167,345 180,345 Z" fill="#D05880" stroke="#903060" stroke-width="1.6"/>
  <path d="M186,356 Q188,367 184,376" fill="none" stroke="#F07898" stroke-width=".8" opacity=".55"/>
  <text x="183" y="368" text-anchor="middle" font-size="7.5" font-weight="600" fill="#620030" font-family="-apple-system,sans-serif">신장</text>
</g>

<!-- 장 -->
<g class="oz" id="oz-intestine" data-part="intestine">
  <path d="M100,345 L98,382 Q98,406 121,408 L149,408 L149,401 M170,345 L172,382 Q172,406 149,408" fill="none" stroke="#D4A060" stroke-width="10" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M110,365 Q119,354 128,365 Q137,376 146,365 Q155,354 162,365 Q168,375 166,385 Q158,394 150,386 Q142,378 134,386 Q126,394 118,386 Q111,378 110,365 Z" fill="#F0C888" stroke="#C09848" stroke-width="1.1"/>
  <text x="135" y="382" text-anchor="middle" font-size="8" font-weight="600" fill="#704808" font-family="-apple-system,sans-serif">장</text>
</g>

<!-- 방광 -->
<g class="oz" id="oz-bladder" data-part="bladder">
  <ellipse cx="135" cy="426" rx="23" ry="18" fill="#88B8E8" stroke="#3870C0" stroke-width="1.6"/>
  <path d="M124,421 Q135,418 146,421" fill="none" stroke="#6898D0" stroke-width=".8" opacity=".7"/>
  <text x="135" y="429" text-anchor="middle" font-size="8" font-weight="600" fill="#183880" font-family="-apple-system,sans-serif">방광</text>
</g>

<!-- 관절 -->
<g class="oz" id="oz-joints" data-part="joints">
  <circle cx="76" cy="164" r="15" fill="#DDDCC0" stroke="#888870" stroke-width="1.5" opacity=".92"/>
  <circle cx="194" cy="164" r="15" fill="#DDDCC0" stroke="#888870" stroke-width="1.5" opacity=".92"/>
  <!-- 무릎 -->
  <circle cx="93" cy="565" r="13" fill="#DDDCC0" stroke="#888870" stroke-width="1.5" opacity=".92"/>
  <circle cx="177" cy="565" r="13" fill="#DDDCC0" stroke="#888870" stroke-width="1.5" opacity=".92"/>
  <text x="76" y="168" text-anchor="middle" font-size="7" font-weight="600" fill="#444422" font-family="-apple-system,sans-serif">관절</text>
  <text x="194" y="168" text-anchor="middle" font-size="7" font-weight="600" fill="#444422" font-family="-apple-system,sans-serif">관절</text>
</g>

<!-- 하지 혈관 -->
<g class="oz" id="oz-legs" data-part="legs">
  <path d="M87,448 Q85,498 87,548 L89,580 L92,630 L95,680" fill="none" stroke="#8090D0" stroke-width="2.5" opacity=".45"/>
  <path d="M183,448 Q185,498 183,548 L181,580 L178,630 L175,680" fill="none" stroke="#8090D0" stroke-width="2.5" opacity=".45"/>
  <path d="M90,448 Q95,498 97,548 L100,590 L103,640" fill="none" stroke="#D04848" stroke-width="1.5" opacity=".35"/>
  <path d="M180,448 Q175,498 173,548 L170,590 L167,640" fill="none" stroke="#D04848" stroke-width="1.5" opacity=".35"/>
</g>

<!-- 전신 오버레이: skin -->
<g class="oz" id="oz-skin" data-part="skin" style="pointer-events:none;opacity:0;">
  <path d="M80,160 L102,154 L135,152 L168,154 L190,160 L194,185 L196,225 L196,318 L193,375 L188,394 L178,428 L188,500 L196,570 L200,640 L204,708 L66,708 L70,640 L74,570 L82,500 L92,428 L82,394 L77,375 L74,318 L74,225 L76,185 Z" fill="rgba(255,90,50,.12)" stroke="rgba(255,70,40,.5)" stroke-width="1.8" stroke-dasharray="9 5"/>
  <ellipse cx="135" cy="69" rx="52" ry="62" fill="rgba(255,90,50,.09)" stroke="rgba(255,70,40,.4)" stroke-width="1.3" stroke-dasharray="7 4"/>
</g>

<!-- 혈액 순환 오버레이 -->
<g class="oz" id="oz-blood" data-part="blood" style="pointer-events:none;opacity:0;">
  <path d="M135,222 Q112,278 108,358 Q104,418 135,436 Q166,418 162,358 Q158,278 135,222 Z" fill="none" stroke="rgba(220,40,40,.4)" stroke-width="2.5" stroke-dasharray="7 4"/>
  <path d="M103,175 L62,228 L58,308 L68,378" fill="none" stroke="rgba(70,100,200,.38)" stroke-width="2" stroke-dasharray="5 4"/>
  <path d="M167,175 L208,228 L212,308 L202,378" fill="none" stroke="rgba(70,100,200,.38)" stroke-width="2" stroke-dasharray="5 4"/>
</g>

<!-- 림프 오버레이 -->
<g class="oz" id="oz-lymph" data-part="lymph" style="pointer-events:none;opacity:0;">
  <circle cx="113" cy="178" r="5.5" fill="none" stroke="rgba(60,180,80,.65)" stroke-width="1.6"/>
  <circle cx="157" cy="178" r="5.5" fill="none" stroke="rgba(60,180,80,.65)" stroke-width="1.6"/>
  <circle cx="103" cy="248" r="4.5" fill="none" stroke="rgba(60,180,80,.55)" stroke-width="1.3"/>
  <circle cx="167" cy="248" r="4.5" fill="none" stroke="rgba(60,180,80,.55)" stroke-width="1.3"/>
  <path d="M113,183 Q107,213 103,252" fill="none" stroke="rgba(60,180,80,.38)" stroke-width="1.2" stroke-dasharray="4 3"/>
  <path d="M157,183 Q163,213 167,252" fill="none" stroke="rgba(60,180,80,.38)" stroke-width="1.2" stroke-dasharray="4 3"/>
</g>

<!-- 면역 오버레이 -->
<g class="oz" id="oz-immune" data-part="immune" style="pointer-events:none;opacity:0;">
  <ellipse cx="135" cy="292" rx="95" ry="130" fill="none" stroke="rgba(80,80,220,.22)" stroke-width="2" stroke-dasharray="9 5"/>
</g>

<!-- 범례 -->
<g font-family="-apple-system,sans-serif">
  <rect x="8" y="718" width="12" height="10" rx="3" fill="#C8D8F4" stroke="#6878C8" stroke-width=".7"/>
  <text x="24" y="727" font-size="8.5" fill="#666">비활성</text>
  <rect x="70" y="718" width="12" height="10" rx="3" fill="#FFD090" stroke="#E09028" stroke-width=".7"/>
  <text x="86" y="727" font-size="8.5" fill="#666">약한 연관</text>
  <rect x="152" y="718" width="12" height="10" rx="3" fill="#FF7050" stroke="#C83010" stroke-width=".7"/>
  <text x="168" y="727" font-size="8.5" fill="#666">강한 연관</text>
</g>

</svg>
<div class="hint" style="font-size:11px;color:#aaa;text-align:center;margin-top:5px;">장기를 클릭하면 상세 정보가 표시됩니다</div>
</div>

<div class="panel-col">
  <div class="info-card" id="defaultCard">
    <div class="info-name">인체 해부도</div>
    <div class="info-sub">장기 또는 신체 부위를 클릭하세요</div>
    <div class="desc" style="margin-top:8px;">왼쪽 인체 그림에서 장기를 클릭하면 해당 부위의 해부학적 기능과 연관 질병 정보를 확인할 수 있습니다.<br><br>빨간색이 진한 부위일수록 현재 선택된 증상과의 연관도가 높습니다.</div>
    <div class="sec-label" style="margin-top:4px;">활성화된 연관 부위</div>
    <div class="tag-wrap" id="activeTags"></div>
  </div>
  <div class="info-card" id="detailCard" style="display:none;">
    <div class="info-name" id="dName"></div>
    <div class="info-sub" id="dSub"></div>
    <div class="int-wrap">
      <div class="int-label">연관도</div>
      <div class="int-track"><div class="int-fill" id="dBar" style="width:0%"></div></div>
      <div class="int-pct" id="dPct">0%</div>
    </div>
    <div class="sec-label">기능·병리</div>
    <div class="desc" id="dDesc"></div>
    <div class="sec-label">연관 예측 질병</div>
    <div class="tag-wrap" id="dDiseases"></div>
    <div style="margin-top:12px;">
      <button onclick="document.getElementById('defaultCard').style.display='block';document.getElementById('detailCard').style.display='none';" style="font-size:11px;color:#888;background:none;border:none;cursor:pointer;padding:0;">← 목록으로</button>
    </div>
  </div>
</div>
</div>

<script>
const PD = """ + part_info_json + """;
const AP = """ + active_json + """;

function getIntColor(pct) {
  if(pct >= 55) return {fill:'#FF7050',stroke:'#CC2810'};
  if(pct >= 30) return {fill:'#FFD090',stroke:'#E09028'};
  if(pct >  0)  return {fill:'#FFE8C0',stroke:'#E0B040'};
  return null;
}
function intensityGradient(pct) {
  if(pct >= 55) return 'linear-gradient(90deg,#f6ad55,#fc8181)';
  if(pct >= 30) return 'linear-gradient(90deg,#68d391,#f6ad55)';
  return '#68d391';
}

// 장기 색상 적용
Object.keys(AP).forEach(part => {
  const pct = Math.round(AP[part]*100);
  const zone = document.getElementById('oz-'+part);
  if(!zone) return;
  const c = getIntColor(pct);
  if(!c) return;
  if(['skin','blood','lymph','immune'].includes(part)) {
    zone.style.opacity = pct > 40 ? '0.88' : pct > 15 ? '0.55' : '0.3';
    zone.style.pointerEvents = 'auto';
    return;
  }
  zone.querySelectorAll('path,ellipse,rect,circle').forEach(el => {
    const of = el.getAttribute('fill');
    if(of && of !== 'none' && !of.startsWith('rgba') && !el.hasAttribute('data-of')) {
      el.setAttribute('data-of', of);
    }
    if(of && of !== 'none' && !of.startsWith('rgba')) {
      el.setAttribute('fill', c.fill);
      el.setAttribute('stroke', c.stroke);
    }
  });
});

// 활성 배지 렌더링
const atEl = document.getElementById('activeTags');
const sorted = Object.keys(AP).sort((a,b)=>AP[b]-AP[a]);
if(sorted.length === 0) {
  atEl.innerHTML = '<span class="tag tag-n">증상을 선택하면 연관 부위가 표시됩니다</span>';
} else {
  sorted.slice(0,12).forEach(p => {
    const pct = Math.round(AP[p]*100);
    const cls = pct>=55?'tag-r':pct>=30?'tag-o':'tag-g';
    const nm = PD[p]?PD[p].name:p;
    atEl.innerHTML += `<span class="tag ${cls}" style="cursor:pointer" onclick="showDetail('${p}')">${nm} ${pct}%</span>`;
  });
}

function showDetail(part) {
  const d = PD[part]; if(!d) return;
  document.getElementById('defaultCard').style.display = 'none';
  document.getElementById('detailCard').style.display  = 'block';
  document.getElementById('dName').textContent = d.name;
  document.getElementById('dSub').textContent  = d.sub || '';
  document.getElementById('dDesc').textContent = d.desc || '';
  const pct = d.intensity || 0;
  const bar = document.getElementById('dBar');
  bar.style.width = pct+'%';
  bar.style.background = intensityGradient(pct);
  document.getElementById('dPct').textContent = pct+'%';
  const dd = document.getElementById('dDiseases');
  dd.innerHTML = '';
  if(d.diseases && d.diseases.length > 0) {
    d.diseases.forEach(x => {
      const cls = x.prob>=30?'tag-r':x.prob>=15?'tag-o':'tag-g';
      dd.innerHTML += `<span class="tag ${cls}">${x.kr} ${x.prob.toFixed(1)}%</span>`;
    });
  } else {
    dd.innerHTML = '<span class="tag tag-n">예측된 질병 없음</span>';
  }
}

document.querySelectorAll('.oz').forEach(el => {
  el.addEventListener('click', () => showDetail(el.getAttribute('data-part')));
});

// 첫 진입 시 가장 강한 연관 부위 자동 표시
if(sorted.length > 0) showDetail(sorted[0]);
</script>
</body></html>"""

    components.html(html, height=780, scrolling=False)



# ════════════════════════════════════════════════════════
#  사이드바
# ════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏥 증상 체크")
    st.caption("해당 증상을 모두 선택하세요")
    st.markdown("**데이터 출처**\n- [Kaggle kaushil268](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning)\n- [Columbia DBMI KB](https://people.dbmi.columbia.edu/~friedma/Projects/DiseaseSymptomKB/index.html)\n- [HPO JAX.org](https://hpo.jax.org/data/annotations)")
    st.divider()
    categories = {
        "전신 증상":["fatigue","weight_loss","weight_gain","lethargy","malaise","restlessness","anxiety","mood_swings"],
        "발열·통증": ["high_fever","mild_fever","headache","joint_pain","back_pain","stomach_pain","chest_pain","abdominal_pain","pain_behind_the_eyes"],
        "피부·외형": ["itching","skin_rash","nodal_skin_eruptions","yellowish_skin","yellowing_of_eyes","dark_urine"],
        "소화기":    ["vomiting","nausea","indigestion","acidity","diarrhoea","constipation","loss_of_appetite"],
        "호흡기":    ["cough","breathlessness","phlegm","congestion","runny_nose","sinus_pressure","throat_irritation","continuous_sneezing"],
        "기타":      ["sweating","chills","shivering","dehydration","blurred_and_distorted_vision","fast_heart_rate","burning_micturition","weakness_in_limbs"],
    }
    selected_symptoms = []
    for cat, syms in categories.items():
        with st.expander(cat, expanded=(cat=="전신 증상")):
            for s in syms:
                if s in SYMPTOM_KR and st.checkbox(SYMPTOM_KR[s], key=s):
                    selected_symptoms.append(s)
    st.divider()
    top_n        = st.slider("상위 N개 질병", 3, 15, 8)
    model_choice = st.radio("예측 모델", ["앙상블 (권장)","Naive Bayes","Random Forest"])

# ════════════════════════════════════════════════════════
#  메인
# ════════════════════════════════════════════════════════
st.title("🩺 증상 기반 질병·인체 시각화 대시보드")
st.caption("ML 확률 예측 · SVG 인체 해부도 · 약품·치료법·민간요법 안내")

nb_model, rf_model, all_syms = train_models()

if not selected_symptoms:
    st.info("👈 왼쪽 사이드바에서 증상을 선택하면 예측 결과와 인체 해부도가 나타납니다.")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("지원 질병","41종"); c2.metric("분석 증상","132개")
    c3.metric("ML 모델","NB+RF 앙상블"); c4.metric("해부도 부위","24개")
    st.stop()

# 예측
input_vec = np.array([[1 if s in selected_symptoms else 0 for s in all_syms]])
nb_p = dict(zip(nb_model.classes_, nb_model.predict_proba(input_vec)[0]))
rf_p = dict(zip(rf_model.classes_, rf_model.predict_proba(input_vec)[0]))
result_rows = []
for d in DISEASE_SYMPTOMS:
    nb_v,rf_v = nb_p.get(d,0),rf_p.get(d,0)
    prob = (nb_v+rf_v)/2 if model_choice=="앙상블 (권장)" else (nb_v if model_choice=="Naive Bayes" else rf_v)
    result_rows.append({"disease":d,"disease_kr":DISEASE_KR.get(d,d),"probability":prob})
result_df = (pd.DataFrame(result_rows).sort_values("probability",ascending=False)
             .head(top_n).reset_index(drop=True))
result_df["prob_pct"] = (result_df["probability"]*100).round(1)

# 신체 부위 활성화 강도
part_intensity: dict = {}
part_disease_map: dict = {}
for _, row in result_df.iterrows():
    d, prob = row["disease"], row["probability"]
    for part in DISEASE_BODY_PARTS.get(d,[]):
        part_intensity[part] = min(1.0, part_intensity.get(part,0) + prob*2)
        part_disease_map.setdefault(part,[]).append({"name":d,"kr":row["disease_kr"],"prob":row["prob_pct"]})
if part_intensity:
    mx = max(part_intensity.values())
    if mx>0: part_intensity = {p:v/mx for p,v in part_intensity.items()}

# 탭
tab1, tab2, tab3 = st.tabs(["📊 예측 결과", "🫀 인체 해부도", "💊 치료법 안내"])

with tab1:
    sym_str = " · ".join(SYMPTOM_KR.get(s,s) for s in selected_symptoms)
    st.markdown(f"**선택된 증상 ({len(selected_symptoms)}개):** {sym_str}")
    col_chart, col_cards = st.columns([3,2])
    with col_chart:
        colors = ["#E24B4A" if p>=30 else "#BA7517" if p>=15 else "#378ADD" for p in result_df["prob_pct"]]
        fig = go.Figure(go.Bar(
            x=result_df["prob_pct"], y=result_df["disease_kr"], orientation="h",
            marker_color=colors, text=[f"{p:.1f}%" for p in result_df["prob_pct"]], textposition="outside",
            hovertemplate="<b>%{y}</b><br>확률: %{x:.1f}%<extra></extra>",
        ))
        fig.update_layout(height=max(300,top_n*44),margin=dict(l=10,r=60,t=10,b=10),
            xaxis_title="가능성 (%)",yaxis=dict(autorange="reversed"),
            plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",font=dict(size=13),
            xaxis=dict(range=[0,min(100,result_df["prob_pct"].max()*1.35)]))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("<span style='color:#E24B4A'>■</span> 높은 가능성 (≥30%)　<span style='color:#BA7517'>■</span> 중간 (15~30%)　<span style='color:#378ADD'>■</span> 낮음 (<15%)", unsafe_allow_html=True)
    with col_cards:
        st.markdown("##### 상위 3개 질병")
        for _, row in result_df.head(3).iterrows():
            p = row["prob_pct"]
            bg,bd,tc = (("#FCEBEB","#E24B4A","#A32D2D") if p>=30 else ("#FAEEDA","#BA7517","#854F0B") if p>=15 else ("#E6F1FB","#378ADD","#185FA5"))
            urg = TREATMENT_DB.get(row["disease"],{}).get("urgency","")
            icon = {"즉시 병원":"🚨","빠른 진료":"⚠️","경과 관찰":"✅"}.get(urg,"")
            parts_kr = [BODY_PART_KR.get(p,"") for p in DISEASE_BODY_PARTS.get(row["disease"],[])[:3]]
            st.markdown(
                f"<div style='background:{bg};border:1px solid {bd};border-radius:10px;padding:.7rem 1rem;margin-bottom:8px;'>"
                f"<div style='font-size:15px;font-weight:500;color:{tc};'>{row['disease_kr']}</div>"
                f"<div style='font-size:11px;color:{tc};opacity:.7;'>{row['disease']}</div>"
                f"<div style='display:flex;justify-content:space-between;align-items:center;margin-top:4px;'>"
                f"<span style='font-size:22px;font-weight:700;color:{tc};'>{p:.1f}%</span>"
                f"<span style='font-size:11px;color:{tc};'>{icon} {urg}</span></div>"
                f"<div style='font-size:11px;color:{tc};opacity:.65;margin-top:2px;'>관련 부위: {' · '.join(parts_kr)}</div>"
                f"</div>", unsafe_allow_html=True)
        with st.expander("📋 전체 결과 테이블"):
            disp=result_df[["disease_kr","disease","prob_pct"]].copy()
            disp.columns=["질병명(한국어)","질병명(영어)","가능성(%)"]
            disp.index=range(1,len(disp)+1)
            st.dataframe(disp,use_container_width=True)

with tab2:
    st.markdown("##### 🫀 인체 해부도 — 예측 질병 연관 부위 시각화")
    st.caption("빨간색이 진할수록 연관도 높음 · 장기를 클릭하면 상세 정보 확인")
    sorted_parts = sorted(part_intensity.items(), key=lambda x:-x[1])
    badge_html = "".join([
        f"<span style='{"background:#FCEBEB;color:#A32D2D;border:1px solid #E24B4A;" if int(v*100)>=60 else "background:#FAEEDA;color:#854F0B;border:1px solid #BA7517;" if int(v*100)>=30 else "background:#EAF3DE;color:#27500A;border:1px solid #3B6D11;"}border-radius:999px;font-size:12px;font-weight:500;padding:3px 11px;margin:2px;display:inline-block;'>{BODY_PART_KR.get(p,p)} {int(v*100)}%</span>"
        for p,v in sorted_parts[:10]
    ])
    if badge_html:
        st.markdown(f"**주요 연관 부위:** {badge_html}", unsafe_allow_html=True)
        st.markdown("")
    render_body_anatomy(part_intensity, part_disease_map)

with tab3:
    st.markdown("##### 💊 질병별 약품·치료법·민간요법")
    st.caption("⚠️ 약품 복용 전 반드시 의사·약사와 상담하세요.")
    top_diseases = result_df["disease"].tolist()
    tab_labels = [f"{DISEASE_KR.get(d,d)} ({result_df.loc[result_df['disease']==d,'prob_pct'].values[0]:.1f}%)" for d in top_diseases]
    d_tabs = st.tabs(tab_labels)
    for dtab,disease in zip(d_tabs,top_diseases):
        with dtab:
            info=TREATMENT_DB.get(disease)
            if not info: st.info("치료 정보 DB 추가 예정"); continue
            urg=info["urgency"]
            icon,bg,bd,tc={"즉시 병원":("🚨","#FCEBEB","#E24B4A","#A32D2D"),"빠른 진료":("⚠️","#FAEEDA","#BA7517","#854F0B"),"경과 관찰":("✅","#EAF3DE","#3B6D11","#27500A")}.get(urg,("ℹ️","#E6F1FB","#185FA5","#0C447C"))
            st.markdown(f"<div style='display:inline-block;background:{bg};border:1px solid {bd};border-radius:8px;padding:4px 16px;font-size:13px;font-weight:500;color:{tc};margin-bottom:14px;'>{icon} {urg}</div>",unsafe_allow_html=True)
            c1,c2,c3=st.columns(3)
            with c1:
                st.markdown("### 💊 추천 약품")
                for drug in info["drugs"]:
                    st.markdown(f"<div style='border:0.5px solid #d0d0d0;border-radius:8px;padding:.6rem .9rem;margin-bottom:8px;'><div style='font-size:14px;font-weight:500;'>{drug['name']}</div><span style='background:#E6F1FB;color:#185FA5;padding:1px 7px;border-radius:4px;font-size:11px;'>{drug['type']}</span><div style='font-size:12px;color:#666;margin-top:4px;'>{drug['note']}</div></div>",unsafe_allow_html=True)
            with c2:
                st.markdown("### 🏥 치료·관리법")
                for i,t in enumerate(info["treatments"],1):
                    st.markdown(f"<div style='display:flex;gap:8px;align-items:flex-start;margin-bottom:7px;'><span style='background:#E1F5EE;color:#0F6E56;border-radius:50%;width:22px;height:22px;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:500;flex-shrink:0;'>{i}</span><span style='font-size:13px;line-height:1.5;'>{t}</span></div>",unsafe_allow_html=True)
            with c3:
                st.markdown("### 🌿 민간요법")
                st.caption("근거 수준이 다양합니다. 보조 수단으로만 활용하세요.")
                for remedy in info["folk_remedies"]:
                    st.markdown(f"<div style='display:flex;gap:8px;align-items:flex-start;margin-bottom:7px;'><span style='color:#3B6D11;font-size:14px;flex-shrink:0;'>🌱</span><span style='font-size:13px;line-height:1.5;'>{remedy}</span></div>",unsafe_allow_html=True)

with st.expander("🔧 데이터·모델 정보"):
    st.markdown("""
**데이터 출처**
| 데이터셋 | 링크 | 용도 |
|---|---|---|
| Kaggle kaushil268 | [링크](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning) | 주 학습 데이터 |
| itachi9604 GitHub | [링크](https://github.com/itachi9604/Disease-Symptom-dataset) | 원본 CSV + severity |
| Columbia DBMI KB | [링크](https://people.dbmi.columbia.edu/~friedma/Projects/DiseaseSymptomKB/index.html) | 임상 NLP 보정 참고 |
| HPO JAX.org | [링크](https://hpo.jax.org/data/annotations) | 희귀질환 확장 참고 |

**모델**: Gaussian Naive Bayes + Random Forest (120 trees) 앙상블  
**인체 해부도**: SVG 직접 구현, 24개 장기/부위 클릭 인터랙션  
**한계**: 공개 데이터 기반으로 실제 임상 정확도와 차이가 있습니다. 반드시 의사 진료를 받으세요.
    """)
