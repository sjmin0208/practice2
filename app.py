import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import streamlit.components.v1 as components
import json, io, os, subprocess, datetime, requests

st.set_page_config(
    page_title="증상 기반 질병·인체 시각화 대시보드",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════
#  다크모드 CSS 주입
# ════════════════════════════════════════════════════════
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

def inject_theme():
    if st.session_state.dark_mode:
        css = """
        <style>
        :root {
            --bg-main: #0e1117;
            --bg-card: #1e2028;
            --bg-sidebar: #161820;
            --text-primary: #e8eaf0;
            --text-secondary: #9ea3b0;
            --border: rgba(255,255,255,0.08);
            --accent: #4f8ef7;
        }
        .stApp { background-color: var(--bg-main) !important; }
        .stApp > header { background-color: var(--bg-main) !important; }
        section[data-testid="stSidebar"] { background-color: var(--bg-sidebar) !important; }
        section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
        .stMarkdown, .stText, p, span, label, div { color: var(--text-primary); }
        .stMetric { background-color: var(--bg-card) !important; border-radius: 10px; padding: 10px; }
        div[data-testid="stExpander"] { background-color: var(--bg-card) !important; border-color: var(--border) !important; }
        .stTabs [data-baseweb="tab-list"] { background-color: var(--bg-card) !important; }
        .stTabs [data-baseweb="tab"] { color: var(--text-secondary) !important; }
        .stTabs [aria-selected="true"] { color: var(--text-primary) !important; }
        div[data-testid="stVerticalBlock"] > div { background-color: transparent !important; }
        .block-container { background-color: var(--bg-main) !important; }
        </style>
        """
    else:
        css = "<style></style>"
    st.markdown(css, unsafe_allow_html=True)

inject_theme()

# ════════════════════════════════════════════════════════
#  한국어 폰트 준비 (PDF용)
# ════════════════════════════════════════════════════════
FONT_REG  = "/home/claude/NotoSansKR-Regular.ttf"
FONT_BOLD = "/home/claude/NotoSansKR-Bold.ttf"

@st.cache_resource
def prepare_fonts():
    """OTF→TTF 변환 (최초 1회)"""
    otf_reg  = "/home/claude/NotoSansKR-Regular.otf"
    otf_bold = "/home/claude/NotoSansKR-Bold.otf"

    def extract_otf(index, ttc_path, out_path):
        if not os.path.exists(out_path):
            from fontTools.ttLib import TTCollection
            col = TTCollection(ttc_path)
            col[index].save(out_path)

    def otf_to_ttf(otf, ttf):
        if not os.path.exists(ttf):
            subprocess.run(["/usr/local/bin/fonttools", "otf2ttf", otf, "-o", ttf],
                           capture_output=True)

    ttc_reg  = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    ttc_bold = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    extract_otf(1, ttc_reg,  otf_reg)
    extract_otf(1, ttc_bold, otf_bold)
    otf_to_ttf(otf_reg,  FONT_REG)
    otf_to_ttf(otf_bold, FONT_BOLD)
    return os.path.exists(FONT_REG) and os.path.exists(FONT_BOLD)

FONTS_OK = prepare_fonts()

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

# ── 질병 상세 설명 (추가)
DISEASE_DESC = {
    "Fungal infection":
        "진균(곰팡이)이 피부·손발톱·점막에 침입해 발생하는 감염입니다. 습하고 통기가 나쁜 환경, 면역력 저하 시 특히 잘 생기며 발무좀·완선·칸디다증 등이 대표적입니다. 접촉 전파가 가능하므로 공용 시설 이용 시 주의가 필요합니다.",
    "Allergy":
        "외부 항원(알레르겐)에 대한 면역계 과민 반응입니다. 꽃가루·집먼지진드기·음식·동물털 등이 원인이며, 히스타민 분비로 재채기·콧물·가려움·두드러기 등이 나타납니다. 아나필락시스(전신 알레르기 반응)는 생명을 위협할 수 있습니다.",
    "GERD":
        "위산이 식도로 역류해 점막을 자극하는 만성 질환입니다. 하부식도괄약근 기능 저하, 비만, 과식, 야식이 주요 원인입니다. 방치하면 바렛식도(전암병변)로 진행될 수 있으므로 꾸준한 관리가 중요합니다.",
    "Chronic cholestasis":
        "담즙 흐름이 만성적으로 차단돼 간과 혈액에 담즙이 축적되는 상태입니다. 원발성 담즙성 경화증·담석·약물 독성이 원인일 수 있으며, 황달·가려움·지방변 등이 나타납니다. 간경화로 진행될 수 있어 조기 진단이 중요합니다.",
    "Drug Reaction":
        "특정 약물에 대한 면역 매개 또는 독성 반응입니다. 피부 발진이 가장 흔하지만 드물게 스티븐스-존슨 증후군처럼 심각한 반응이 생길 수 있습니다. 원인 약물 식별과 즉각적인 중단이 핵심 치료입니다.",
    "Peptic ulcer disease":
        "위나 십이지장 점막에 궤양(패인 상처)이 생기는 질환입니다. H.pylori균 감염(70%)과 NSAIDs 장기 복용이 주요 원인입니다. 복통·속쓰림이 대표 증상이며 출혈·천공 등 합병증 발생 시 응급 처치가 필요합니다.",
    "AIDS":
        "HIV 바이러스가 CD4 T림프구를 파괴해 면역 기능이 심각하게 저하된 상태입니다. 초기 감염 후 수년간 무증상 기간을 거쳐 면역 결핍이 진행됩니다. 조기 항레트로바이러스 치료(ART)로 정상에 가까운 수명과 삶의 질을 유지할 수 있습니다.",
    "Diabetes":
        "인슐린 분비 부족(1형) 또는 인슐린 저항성 증가(2형)로 혈당이 만성적으로 높아지는 대사 질환입니다. 장기 합병증으로 망막병증·신부전·말초신경병증·심혈관 질환이 발생합니다. 한국인은 서양인 대비 낮은 체중에서도 2형 당뇨가 발생할 수 있습니다.",
    "Gastroenteritis":
        "바이러스·세균·기생충에 의한 위장관 염증으로 구토와 설사가 주증상입니다. 노로바이러스가 가장 흔한 원인이며 오염된 음식·물이 매개됩니다. 대부분 자연 회복되나 탈수 예방이 가장 중요하며 영아·노인은 입원이 필요할 수 있습니다.",
    "Bronchial Asthma":
        "기도의 만성 염증과 과민 반응으로 기도가 좁아져 호흡 곤란·천명음이 반복되는 질환입니다. 알레르겐·운동·찬 공기·스트레스가 발작을 유발합니다. 흡입 스테로이드로 염증을 조절하고 속효성 기관지확장제로 급성 발작을 치료합니다.",
    "Hypertension":
        "수축기 혈압 140mmHg 이상 또는 이완기 90mmHg 이상이 지속되는 상태입니다. '침묵의 살인자'라 불릴 만큼 증상이 없다가 뇌졸중·심근경색·신부전으로 이어집니다. 나트륨 제한, 규칙적 운동, 체중 관리가 비약물 치료의 핵심입니다.",
    "Migraine":
        "두개 내 혈관 수축·확장과 삼차신경 자극이 복합적으로 작용하는 신경혈관성 두통입니다. 박동성 편측 두통, 오심, 광선·소리 공포증이 특징이며 전조 증상(시각 이상)이 먼저 나타나기도 합니다. 여성에서 3배 많이 발생하며 월경 주기와 연관됩니다.",
    "Cervical spondylosis":
        "경추 추간판과 관절의 퇴행성 변화로 신경근이 압박되는 질환입니다. 장시간 컴퓨터·스마트폰 사용이 위험 요인이며 목·어깨 통증, 팔 저림이 나타납니다. 심한 경우 척수 압박으로 사지 마비가 올 수 있어 주의가 필요합니다.",
    "Paralysis (brain hemorrhage)":
        "뇌혈관 파열로 혈액이 뇌 조직에 유입되어 신경 손상이 발생하는 응급 질환입니다. 갑작스러운 편측 마비·언어 장애·두통이 특징입니다. FAST(얼굴 비대칭·팔 처짐·언어 이상·즉시 신고) 원칙으로 즉각 대응해야 합니다.",
    "Jaundice":
        "혈중 빌리루빈이 증가해 피부·눈 흰자가 노랗게 변하는 증상입니다. 원인은 간세포 손상(간염), 담즙 폐쇄(담석), 용혈성 빈혈 등 다양합니다. 증상 자체보다 원인 질환 파악이 핵심이며, 진한 소변·탈색 변을 동반합니다.",
    "Malaria":
        "열대열 원충 등 플라스모디움 기생충이 모기를 통해 전파되는 감염병입니다. 주기적 고열·오한·발한 삼연증이 특징이며 열대열 말라리아는 뇌 침범 시 생명을 위협합니다. 해외 여행 전 예방약 복용과 모기 기피가 필수입니다.",
    "Chicken pox":
        "수두-대상포진 바이러스(VZV)의 초감염으로 발생하는 고도 전염성 질환입니다. 가려운 수포성 발진이 전신에 퍼지며 평생 면역을 남깁니다. 바이러스는 척수신경절에 잠복해 노년기에 대상포진으로 재활성화될 수 있습니다.",
    "Dengue":
        "이집트숲모기가 전파하는 뎅기 바이러스 감염병입니다. 고열·심한 근육통·발진이 나타나며 '뼈 부수는 병'으로도 불립니다. 혈소판 감소와 혈관 투과성 증가로 출혈열로 진행 가능하며 이부프로펜·아스피린은 금기입니다.",
    "Typhoid":
        "살모넬라 타이피균이 오염된 음식·물을 통해 전파되는 소화기 감염병입니다. 지속적 고열·복통·서맥이 특징이며 장천공이라는 치명적 합병증이 생길 수 있습니다. 위생적 식수와 위생 관리가 예방의 핵심입니다.",
    "hepatitis A":
        "A형 간염 바이러스(HAV)가 분변-구강 경로로 전파되는 급성 간염입니다. 대부분 자연 회복되나 고령자·만성 간질환자는 중증화 위험이 있습니다. 백신으로 95% 예방 가능하며 한국은 40세 이하에서 항체 보유율이 낮아 접종이 권장됩니다.",
    "Hepatitis B":
        "B형 간염 바이러스(HBV)가 혈액·성 접촉·수직 감염으로 전파되는 바이러스성 간염입니다. 만성화되면 간경화·간암으로 이어질 수 있으며 한국인 간암의 약 70%가 B형 간염과 연관됩니다. 신생아 예방접종으로 수직 감염을 90% 이상 차단할 수 있습니다.",
    "Hepatitis C":
        "C형 간염 바이러스(HCV) 감염으로 주로 혈액 접촉을 통해 전파됩니다. 70~85%가 만성화되어 간경화·간암 위험이 높습니다. 백신은 없지만 DAA(직접 작용 항바이러스제)로 8~12주 치료 시 완치율이 95% 이상입니다.",
    "Hepatitis D":
        "D형 간염 바이러스(HDV)는 B형 간염 바이러스가 있어야만 감염되는 위성 바이러스입니다. B형+D형 동시 감염 또는 B형 환자의 중복 감염으로 간부전 위험이 높아집니다. B형 간염 예방접종이 D형 간염도 동시에 예방합니다.",
    "Hepatitis E":
        "E형 간염 바이러스(HEV)가 오염된 물·덜 익힌 돼지고기를 통해 전파되는 간염입니다. 대부분 자연 회복되나 임산부에서 전격성 간부전으로 사망률이 15~25%에 달합니다. 개발도상국 여행 시 위생적 식수 관리가 중요합니다.",
    "Alcoholic hepatitis":
        "과도한 알코올 섭취로 간세포가 손상되는 급성 염증 반응입니다. 간비대·황달·복수·간성 뇌증이 나타나며 중증인 경우 사망률이 30~50%에 달합니다. 완전 금주만이 유일한 치료이며 간이식이 필요할 수 있습니다.",
    "Tuberculosis":
        "결핵균(M. tuberculosis)이 비말로 전파되는 만성 감염병입니다. 폐에서 주로 발생하나 림프절·뼈·신장 등 전신을 침범할 수 있습니다. 한국은 OECD 국가 중 결핵 발생률이 높으며, 최소 6개월 이상의 복합 항결핵 치료가 필수입니다.",
    "Common Cold":
        "200여 종의 바이러스(주로 리노바이러스)에 의한 상기도 감염입니다. 재채기·콧물·인후통이 주증상이며 대부분 7~10일 내 자연 회복됩니다. 항생제는 무효하며 충분한 수분·휴식이 기본 치료입니다.",
    "Pneumonia":
        "세균·바이러스·진균이 폐 실질을 침범해 폐포에 삼출물이 차는 감염성 질환입니다. 고열·기침·가래·호흡 곤란이 주증상이며 중증도 평가(PSI, CURB-65)에 따라 입원 여부를 결정합니다. 폐렴구균 백신으로 주요 세균성 폐렴을 예방할 수 있습니다.",
    "Dimorphic hemorrhoids(piles)":
        "항문 주위 정맥총이 팽창하거나 항문 점막이 탈출하는 질환입니다. 변비로 인한 복압 증가, 장시간 좌식 생활이 주요 원인입니다. 출혈·통증·탈출이 주증상이며 섬유질 식이와 좌욕이 증상 완화에 효과적입니다.",
    "Heart attack":
        "관상동맥이 혈전으로 막혀 심근이 괴사하는 응급 질환입니다. 가슴 압박감·쥐어짜는 흉통·팔·턱으로 방사통이 특징이며 여성은 비전형적 증상(피로·소화불량)으로 나타날 수 있습니다. 발생 후 90분 이내 재관류 치료가 예후를 결정합니다.",
    "Varicose veins":
        "다리의 표재성 정맥 판막이 기능을 잃어 혈액이 역류하며 혈관이 팽창·구불구불해지는 질환입니다. 오랜 서 있기·임신·비만이 위험 요인이며 다리 무거움·부종·야간 경련이 나타납니다. 미용 문제를 넘어 혈전·피부 궤양으로 진행될 수 있습니다.",
    "Hypothyroidism":
        "갑상선 호르몬(T3/T4) 분비 부족으로 전신 대사가 저하되는 질환입니다. 피로·체중 증가·추위 불내성·변비·무기력이 특징입니다. 자가면역 갑상선염(하시모토병)이 가장 흔한 원인이며 레보티록신 보충으로 효과적으로 조절됩니다.",
    "Hyperthyroidism":
        "갑상선 호르몬 과잉으로 전신 대사가 항진되는 질환입니다. 체중 감소·빠른 심박·더위 불내성·손 떨림·불안이 특징입니다. 그레이브스병이 가장 흔한 원인이며 미치료 시 갑상선 위기(thyroid storm)라는 생명 위협 상황이 발생할 수 있습니다.",
    "Hypoglycemia":
        "혈당이 70mg/dL 미만으로 떨어지는 상태입니다. 당뇨 약물 과용량·식사 거름·과도한 운동이 원인이며 떨림·식은땀·어지럼·의식 저하가 나타납니다. 심한 경우 경련·혼수로 이어지므로 즉각적인 포도당 섭취가 필요합니다.",
    "Osteoarthritis":
        "관절 연골이 점진적으로 마모되어 통증·강직·기능 저하가 발생하는 퇴행성 관절 질환입니다. 무릎·고관절·손가락이 주요 이환 부위이며, 비만이 가장 강력한 위험 인자입니다. 완치는 어렵지만 체중 감량·저충격 운동으로 진행을 늦출 수 있습니다.",
    "Arthritis":
        "관절의 염증성 질환의 총칭으로, 류마티스 관절염은 활액막을 면역계가 공격하는 자가면역 질환입니다. 양측성 대칭 관절염·조조 강직·전신 피로가 특징입니다. 조기 DMARD 치료로 관절 파괴를 예방하는 것이 가장 중요합니다.",
    "(vertigo) Paroxysmal Positional Vertigo":
        "내이의 이석(탄산칼슘 결정)이 반고리관으로 이탈해 특정 자세에서 강한 회전성 어지럼증이 유발되는 질환입니다. 누웠다 일어날 때나 고개를 돌릴 때 수 초~수십 초의 어지럼증이 발생합니다. 엡리 이석 정복술(Epley maneuver)로 80~90% 완치가 가능합니다.",
    "Acne":
        "피지선 과분비·각질 폐색·여드름균(C. acnes) 증식·염증 반응이 복합적으로 작용하는 만성 피부 질환입니다. 10~20대에 가장 흔하며 호르몬 변화와 밀접합니다. 흉터 예방을 위해 스스로 짜지 않고 적절한 치료를 받는 것이 중요합니다.",
    "Urinary tract infection":
        "대장균(80%)이 요도를 통해 방광으로 상행 감염되는 질환입니다. 여성은 요도가 짧아(4cm) 남성보다 40배 높은 빈도로 발생합니다. 배뇨통·잔뇨감·빈뇨가 주증상이며 신장까지 상행하면 고열·옆구리 통증을 동반하는 신우신염으로 악화됩니다.",
    "Psoriasis":
        "T세포 과활성화로 피부 세포가 비정상적으로 빠르게 증식하는 만성 자가면역 피부 질환입니다. 은백색 인설이 덮인 홍반성 판이 특징적이며 두피·팔꿈치·무릎에 호발합니다. 30%에서 건선 관절염이 동반되며 심혈관 질환·대사증후군 위험도 높습니다.",
    "Impetigo":
        "황색포도상구균 또는 화농성 연쇄상구균이 표피를 침범하는 고도 전염성 피부 감염입니다. 주로 소아의 얼굴·사지에 꿀색 딱지를 형성하며 직접 접촉으로 빠르게 퍼집니다. 항생제 치료에 잘 반응하지만 치료 전까지는 격리가 필요합니다.",
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
#  AI 한줄 평가
# ════════════════════════════════════════════════════════
def get_ai_comment(symptoms_kr: list, top_diseases: list, gender: str, age_group: str) -> str:
    """Anthropic API로 AI 한줄 평가 생성"""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        except Exception:
            api_key = ""
    if not api_key:
        return "⚠️ ANTHROPIC_API_KEY가 설정되지 않았습니다. Streamlit Secrets 또는 환경변수에 키를 추가해주세요."

    sym_str = ", ".join(symptoms_kr[:10])
    dis_str = " / ".join([f"{d['kr']} {d['pct']:.1f}%" for d in top_diseases[:3]])
    demo = f"{gender}성 {age_group}" if gender != "선택 안 함" else f"{age_group}"

    prompt = f"""당신은 경험 많은 한국 내과 전문의입니다. 아래 정보를 보고 환자에게 해줄 수 있는 친절하고 실용적인 한줄 코멘트를 작성해주세요.

환자 정보: {demo}
주요 증상: {sym_str}
AI 예측 상위 질병: {dis_str}

요구사항:
- 반드시 한국어로 작성
- 2~3문장 이내의 간결한 의견
- 긴급도에 따라 병원 방문 권고 포함
- 공감적이고 따뜻한 어조
- 절대로 확정적 진단을 내리지 말 것
- 면책 문구 불필요 (UI에 이미 있음)"""

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 300,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=15,
        )
        data = resp.json()
        if "content" in data and data["content"]:
            return data["content"][0]["text"].strip()
        return f"API 응답 오류: {data.get('error', {}).get('message', '알 수 없는 오류')}"
    except Exception as e:
        return f"연결 오류: {str(e)}"

# ════════════════════════════════════════════════════════
#  PDF 리포트 생성
# ════════════════════════════════════════════════════════
def generate_pdf_report(
    selected_symptoms, result_df, part_intensity,
    gender, age_group, model_choice, ai_comment=""
) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether
    )
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    # 폰트 등록
    pdfmetrics.registerFont(TTFont("NotoKR",     FONT_REG))
    pdfmetrics.registerFont(TTFont("NotoKR-Bold",FONT_BOLD))

    # 스타일 정의
    def S(name, **kw):
        defaults = dict(fontName="NotoKR", fontSize=10, leading=16, spaceAfter=4)
        defaults.update(kw)          # kw가 defaults를 덮어씀 (fontName 포함)
        return ParagraphStyle(name, **defaults)

    sTitle   = S("T", fontName="NotoKR-Bold", fontSize=20, leading=28, spaceAfter=6, textColor=colors.HexColor("#1a1a2e"))
    sH1      = S("H1",fontName="NotoKR-Bold", fontSize=14, leading=20, spaceBefore=14, spaceAfter=4, textColor=colors.HexColor("#1e3a5f"))
    sH2      = S("H2",fontName="NotoKR-Bold", fontSize=11, leading=16, spaceBefore=8,  spaceAfter=2, textColor=colors.HexColor("#2c5282"))
    sBody    = S("B",  fontSize=9,  leading=15, textColor=colors.HexColor("#333333"))
    sSmall   = S("Sm", fontSize=8,  leading=13, textColor=colors.HexColor("#666666"))
    sCaption = S("Ca", fontSize=8,  leading=12, textColor=colors.HexColor("#888888"), fontName="NotoKR")
    sAI      = S("AI", fontSize=9,  leading=16, textColor=colors.HexColor("#1a4731"),
                 backColor=colors.HexColor("#e8f5e9"), borderPadding=8, borderWidth=1,
                 borderColor=colors.HexColor("#81c784"))
    sWarn    = S("W",  fontSize=7.5,leading=12, textColor=colors.HexColor("#7f4f24"),
                 fontName="NotoKR")

    # 긴급도 색상
    URG_COLOR = {"즉시 병원": "#c62828", "빠른 진료": "#e65100", "경과 관찰": "#2e7d32"}

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=18*mm, bottomMargin=18*mm,
        title="증상 기반 질병 예측 리포트",
        author="Self-Check Dashboard",
    )
    story = []
    now = datetime.datetime.now().strftime("%Y년 %m월 %d일 %H:%M")

    # ── 헤더
    story.append(Paragraph("🩺 증상 기반 질병 예측 리포트", sTitle))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1e3a5f"), spaceAfter=6))
    demo = f"{gender}성 · {age_group}" if gender != "선택 안 함" else age_group
    story.append(Paragraph(f"생성일시: {now}　|　대상: {demo}　|　모델: {model_choice}", sSmall))
    story.append(Spacer(1, 6))

    # ── 면책 조항
    story.append(Paragraph(
        "⚠️ 본 리포트는 AI 기반 참고 자료로, 의료 진단을 대체하지 않습니다. "
        "정확한 진단과 치료를 위해 반드시 의사와 상담하세요.", sWarn))
    story.append(Spacer(1, 10))

    # ── 선택 증상
    story.append(Paragraph("1. 선택한 증상", sH1))
    sym_kr_list = [SYMPTOM_KR.get(s, s) for s in selected_symptoms]
    sym_text = "  ·  ".join(sym_kr_list) if sym_kr_list else "없음"
    story.append(Paragraph(sym_text, sBody))
    story.append(Spacer(1, 8))

    # ── AI 한줄평가
    if ai_comment and not ai_comment.startswith("⚠️") and not ai_comment.startswith("API") and not ai_comment.startswith("연결"):
        story.append(Paragraph("2. AI 한줄 평가", sH1))
        story.append(Paragraph(f"🤖 {ai_comment}", sAI))
        story.append(Spacer(1, 8))
        next_sec = "3"
    else:
        next_sec = "2"

    # ── 예측 결과 테이블
    story.append(Paragraph(f"{next_sec}. 질병 예측 결과 (상위 {len(result_df)}개)", sH1))
    next_sec = str(int(next_sec) + 1)

    tbl_data = [["순위", "질병명 (한국어)", "질병명 (영어)", "가능성(%)", "긴급도"]]
    for i, row in result_df.iterrows():
        urg = TREATMENT_DB.get(row["disease"], {}).get("urgency", "경과 관찰")
        tbl_data.append([
            str(i+1),
            row["disease_kr"],
            row["disease"],
            f"{row['prob_pct']:.1f}%",
            urg,
        ])

    col_w = [10*mm, 42*mm, 52*mm, 18*mm, 22*mm]
    tbl = Table(tbl_data, colWidths=col_w, repeatRows=1)
    tbl_style = [
        ("FONTNAME",     (0,0), (-1,-1), "NotoKR"),
        ("FONTNAME",     (0,0), (-1, 0), "NotoKR-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 8),
        ("FONTSIZE",     (0,0), (-1, 0), 9),
        ("BACKGROUND",   (0,0), (-1, 0), colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR",    (0,0), (-1, 0), colors.white),
        ("ALIGN",        (0,0), (-1,-1), "CENTER"),
        ("ALIGN",        (1,1), (2,-1), "LEFT"),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f8f9fa"), colors.white]),
        ("GRID",         (0,0), (-1,-1), 0.4, colors.HexColor("#dee2e6")),
        ("TOPPADDING",   (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
    ]
    # 긴급도 색상 적용
    for i, row in result_df.iterrows():
        urg = TREATMENT_DB.get(row["disease"], {}).get("urgency", "경과 관찰")
        c = URG_COLOR.get(urg, "#333333")
        tbl_style.append(("TEXTCOLOR", (4, i+1), (4, i+1), colors.HexColor(c)))
        tbl_style.append(("FONTNAME",  (4, i+1), (4, i+1), "NotoKR-Bold"))
    tbl.setStyle(TableStyle(tbl_style))
    story.append(tbl)
    story.append(Spacer(1, 10))

    # ── 주요 연관 부위
    story.append(Paragraph(f"{next_sec}. 주요 연관 신체 부위", sH1))
    next_sec = str(int(next_sec) + 1)
    sorted_parts = sorted(part_intensity.items(), key=lambda x: -x[1])[:8]
    parts_text = "  |  ".join([f"{BODY_PART_KR.get(p,p)} ({int(v*100)}%)" for p,v in sorted_parts])
    story.append(Paragraph(parts_text, sBody))
    story.append(Spacer(1, 10))

    # ── 질병별 상세 (상위 5개)
    story.append(Paragraph(f"{next_sec}. 주요 질병 상세 정보", sH1))
    next_sec = str(int(next_sec) + 1)

    for _, row in result_df.head(5).iterrows():
        d = row["disease"]
        dk = row["disease_kr"]
        p = row["prob_pct"]
        info = TREATMENT_DB.get(d, {})
        urg  = info.get("urgency", "경과 관찰")
        desc = DISEASE_DESC.get(d, "")
        urg_c = colors.HexColor(URG_COLOR.get(urg, "#333333"))

        block = []
        # 질병명 헤더
        block.append(Paragraph(
            f"<font color='#{URG_COLOR.get(urg,'333333')[1:]}'><b>{dk}</b></font>"
            f"  <font size='8'>{d}</font>"
            f"  <font color='#888888' size='8'>가능성 {p:.1f}%  |  {urg}</font>",
            S("DH", fontName="NotoKR-Bold", fontSize=11, leading=16, spaceBefore=6, spaceAfter=2)
        ))
        if desc:
            block.append(Paragraph(desc, sSmall))
        # 추천 약품
        if info.get("drugs"):
            block.append(Paragraph("▸ 추천 약품", S("SH", fontName="NotoKR-Bold", fontSize=8.5, leading=14, textColor=colors.HexColor("#1a237e"))))
            drug_text = "  ·  ".join([f"{dg['name']} ({dg['type']})" for dg in info["drugs"]])
            block.append(Paragraph(drug_text, sSmall))
        # 치료법
        if info.get("treatments"):
            block.append(Paragraph("▸ 관리·치료", S("SH", fontName="NotoKR-Bold", fontSize=8.5, leading=14, textColor=colors.HexColor("#1a237e"))))
            block.append(Paragraph("  ·  ".join(info["treatments"]), sSmall))
        block.append(Spacer(1, 4))
        story.append(KeepTogether(block))

    # ── 푸터 구분선
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc"), spaceBefore=10, spaceAfter=6))
    story.append(Paragraph(
        "본 리포트는 참고용이며 실제 의료 진단을 대체하지 않습니다. "
        "증상이 지속되거나 악화되면 반드시 전문 의료기관을 방문하세요. "
        f"생성: Self-Check Dashboard  |  {now}",
        sCaption
    ))

    doc.build(story)
    return buf.getvalue()

# ════════════════════════════════════════════════════════
#  인체 해부도 (v6 — 전신계 버튼 분리)
# ════════════════════════════════════════════════════════
BODY_PART_INFO = {
    "brain":       {"name":"뇌",        "sub":"중추신경계 · 두개골 내부",  "desc":"약 1,000억 개 신경세포로 이루어진 인체 제어 센터입니다. 편두통은 뇌혈관 수축·확장 이상, 뇌출혈은 혈관 파열로 뇌 조직이 손상되는 응급 상황입니다."},
    "heart":       {"name":"심장",      "sub":"흉강 좌측 · 근육 펌프",     "desc":"하루 약 10만 번 박동합니다. 심근경색은 관상동맥 폐색으로 심근이 괴사하는 응급 상황이며 90분 내 치료가 핵심입니다."},
    "lungs":       {"name":"폐",        "sub":"흉강 양측 · 가스 교환",     "desc":"좌폐 2엽·우폐 3엽, 폐포 면적 약 70㎡입니다. 폐렴은 폐포를 세균·바이러스가 침범하고, 결핵균은 폐 상엽을 주로 침범합니다."},
    "liver":       {"name":"간",        "sub":"복강 우상부 · 해독·대사",   "desc":"체내 최대 장기(1.2~1.5kg)로 500가지 이상의 대사 기능을 담당합니다. A~E형 간염 모두 간세포를 표적으로 하며 만성 염증 반복 시 간경화로 진행됩니다."},
    "stomach":     {"name":"위",        "sub":"복강 좌상부 · 소화 시작",   "desc":"강산성 위액(pH 1~3)으로 음식을 분해합니다. H.pylori균이 점막 보호층을 파괴해 소화성 궤양을 유발하고, GERD는 하부식도괄약근 기능 이상으로 발생합니다."},
    "intestine":   {"name":"소장·대장", "sub":"복강 중앙 · 흡수·배설",    "desc":"소장(6~7m)은 영양소를, 대장은 수분을 흡수합니다. 장티푸스균은 소장 림프절을 침범하고, 치질은 항문 주위 정맥총 팽창으로 발생합니다."},
    "kidney":      {"name":"신장",      "sub":"후복막 좌우 · 혈액 정화",   "desc":"하루 약 180L 혈액을 여과해 1~2L 소변을 생성합니다. 당뇨·고혈압의 장기 합병증으로 신기능이 저하되며, 요로 감염은 대장균의 상행 감염으로 주로 발생합니다."},
    "skin":        {"name":"피부",      "sub":"전신 표면 · 1차 방어막",    "desc":"체표면적 약 1.5~2㎡의 인체 최대 기관입니다. 건선은 T세포 과활성화로 피부세포가 과증식하고, 여드름은 피지선 과분비와 세균 감염으로 발생합니다."},
    "joints":      {"name":"관절",      "sub":"뼈 연결부 · 운동 담당",     "desc":"초자연골·활액막·인대·건으로 구성됩니다. 골관절염은 연골 마모, 류마티스 관절염은 활액막을 자가면역이 공격하는 전신 염증 질환입니다."},
    "thyroid":     {"name":"갑상선",    "sub":"경부 전면 · 호르몬 분비",   "desc":"T3·T4 호르몬을 분비합니다. 기능 항진은 체중 감소·빠른 심박, 기능 저하는 피로·체중 증가 증상으로 나타납니다."},
    "pancreas":    {"name":"췌장",      "sub":"복강 후방 · 혈당 조절",     "desc":"랑게르한스섬의 β세포가 인슐린을 분비합니다. 1형 당뇨는 β세포 파괴, 2형 당뇨는 인슐린 저항성 증가가 핵심 기전입니다."},
    "gallbladder": {"name":"담낭",      "sub":"간 하면 · 담즙 저장",       "desc":"담즙을 농축·저장했다가 지방 소화 시 분비합니다. 담즙 성분 불균형 시 담석이 형성되고, 담즙 역류 시 황달이 발생합니다."},
    "spleen":      {"name":"비장",      "sub":"복강 좌상부 · 면역 필터",   "desc":"노화된 적혈구를 제거하고 면역세포를 생산합니다. 말라리아 감염 시 감염된 적혈구가 집적되어 비장 비대가 특징적으로 나타납니다."},
    "bladder":     {"name":"방광",      "sub":"골반강 · 소변 저장",        "desc":"400~600ml의 소변을 저장합니다. 요로 감염의 80%는 대장균이 요도를 통해 상행 감염되며, 여성은 요도가 짧아 감염 위험이 높습니다."},
    "spine":       {"name":"척추",      "sub":"중심축 · 신경 보호",        "desc":"경추 7개·흉추 12개·요추 5개로 구성됩니다. 경추 척추증은 추간판 변성으로 신경근이 압박되어 목·어깨·팔로 방사통이 발생합니다."},
    "legs":        {"name":"하지·정맥", "sub":"하체 · 혈액 환류",          "desc":"하지 정맥판막이 일방통행 밸브 역할을 합니다. 정맥류는 판막 기능 부전으로 혈액이 역류하면서 정맥벽이 팽창하는 질환입니다."},
    "eye":         {"name":"눈",        "sub":"시각기관 · 안구",            "desc":"편두통 전조 증상(광선 공포증·섬광)이 나타납니다. 당뇨 합병증으로 망막 미세혈관이 손상되는 망막병증은 성인 실명의 주요 원인입니다."},
    "ear":         {"name":"귀 (내이)", "sub":"청각·전정기관",             "desc":"내이 이석이 반고리관으로 이탈하면 특정 자세에서 강한 회전성 어지럼증(이석증)이 유발됩니다."},
    "esophagus":   {"name":"식도",      "sub":"인두~위 연결 통로",          "desc":"하부식도괄약근이 위산 역류를 막습니다. GERD에서 반복적인 위산 자극은 바렛 식도로 진행될 수 있습니다."},
    "blood":       {"name":"혈액",      "sub":"전신 순환 · 운반 매체",      "desc":"적혈구·백혈구·혈소판·혈장으로 구성됩니다. 말라리아 원충은 적혈구 내 증식·파열하고, 뎅기열은 혈소판을 감소시켜 출혈 경향을 높입니다."},
    "lymph":       {"name":"림프계",    "sub":"면역 네트워크",              "desc":"림프관·림프절·비장·흉선으로 구성됩니다. 감염 시 림프절에서 T·B세포가 활성화되며, AIDS에서는 HIV가 CD4 T세포를 직접 파괴합니다."},
    "immune":      {"name":"면역계",    "sub":"전신 방어 체계",             "desc":"선천면역과 후천면역으로 구성됩니다. HIV는 CD4 T세포를 점진적으로 파괴하고, CD4 수치 200/μL 미만이면 AIDS로 진행됩니다."},
    "neck":        {"name":"경추",      "sub":"목 척추",                   "desc":"경추 7개 척추뼈가 뇌와 몸통을 연결합니다. 추간판 변성으로 신경근이 압박되면 목·어깨·팔 방사통이 발생합니다."},
    "nose":        {"name":"코",        "sub":"호흡·후각기관",             "desc":"비강 점막이 공기를 가온·가습·여과합니다. 알레르기 비염은 알레르겐에 반응한 비만세포가 히스타민을 분비해 콧물·재채기를 유발합니다."},
}
SYSTEMIC_PARTS = ["blood", "skin", "lymph", "immune"]


def render_body_anatomy(active_parts: dict, part_disease_map: dict):
    part_data_js = {}
    for part, info in BODY_PART_INFO.items():
        diseases = sorted(part_disease_map.get(part, []), key=lambda x: -x["prob"])[:6]
        part_data_js[part] = {
            "name":      info["name"],
            "sub":       info["sub"],
            "desc":      info["desc"],
            "intensity": round(active_parts.get(part, 0) * 100),
            "diseases":  diseases,
        }

    active_json    = json.dumps({p: round(v, 3) for p, v in active_parts.items()})
    part_data_json = json.dumps(part_data_js, ensure_ascii=False)
    systemic_json  = json.dumps(SYSTEMIC_PARTS)

    dark = "true" if st.session_state.dark_mode else "false"

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:transparent;}}
.layout{{display:flex;gap:16px;align-items:flex-start;}}
.svg-col{{flex:0 0 260px;}}
.panel-col{{flex:1;min-width:0;display:flex;flex-direction:column;gap:10px;}}
.ob{{cursor:pointer;}}
.ob:hover > *{{opacity:.75;}}
.card{{background:var(--c-card,#fff);border:.5px solid var(--c-border,rgba(0,0,0,.1));border-radius:12px;padding:16px;}}
.card-name{{font-size:17px;font-weight:500;color:var(--c-text,#111);margin-bottom:2px;}}
.card-sub{{font-size:11px;color:var(--c-sub,#888);margin-bottom:12px;}}
.bar-row{{display:flex;align-items:center;gap:8px;margin-bottom:12px;}}
.bar-lbl{{font-size:10px;font-weight:600;color:var(--c-sub,#999);text-transform:uppercase;letter-spacing:.06em;min-width:40px;}}
.bar-track{{flex:1;height:5px;background:var(--c-track,#eee);border-radius:3px;overflow:hidden;}}
.bar-fill{{height:100%;border-radius:3px;transition:width .4s,background .3s;}}
.bar-pct{{font-size:11px;font-weight:500;min-width:28px;text-align:right;color:var(--c-text,#111);}}
.sec{{font-size:10px;font-weight:600;color:var(--c-sub,#999);text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px;}}
.desc{{font-size:12px;color:var(--c-sub,#555);line-height:1.65;margin-bottom:12px;}}
.tags{{display:flex;flex-wrap:wrap;gap:4px;}}
.tag{{font-size:11px;padding:2px 9px;border-radius:999px;font-weight:500;border:1px solid;cursor:default;}}
.tag-r{{background:#FCEBEB;color:#791F1F;border-color:#F09595;}}
.tag-o{{background:#FAEEDA;color:#633806;border-color:#EF9F27;}}
.tag-g{{background:#EAF3DE;color:#27500A;border-color:#97C459;}}
.tag-n{{background:var(--c-badge,#f5f5f3);color:var(--c-sub,#888);border-color:var(--c-border,rgba(0,0,0,.1));}}
.back-btn{{font-size:11px;color:var(--c-sub,#888);background:none;border:none;cursor:pointer;padding:0;margin-top:10px;display:block;}}
.hint{{font-size:10px;color:#bbb;text-align:center;margin-top:5px;}}
.systemic-grid{{display:grid;grid-template-columns:1fr 1fr;gap:7px;}}
.sys-btn{{display:flex;align-items:center;gap:8px;padding:8px 11px;border-radius:9px;border:1.8px solid;cursor:pointer;transition:opacity .15s,transform .1s;text-align:left;}}
.sys-btn:hover{{opacity:.75;transform:scale(1.02);}}
.sys-dot{{width:11px;height:11px;border-radius:50%;flex-shrink:0;}}
.sys-name{{font-size:12px;font-weight:600;}}
.sys-pct{{font-size:11px;margin-left:auto;font-weight:500;}}
/* 다크모드 */
body.dark{{--c-card:#1e2028;--c-border:rgba(255,255,255,.1);--c-text:#e8eaf0;--c-sub:#9ea3b0;--c-track:#333;--c-badge:#2a2a28;}}
body.dark .tag-r{{background:#501313;color:#F7C1C1;border-color:#A32D2D;}}
body.dark .tag-o{{background:#412402;color:#FAC775;border-color:#854F0B;}}
body.dark .tag-g{{background:#173404;color:#C0DD97;border-color:#3B6D11;}}
body.dark .sys-btn{{background:#1e2028 !important;}}
</style></head>
<body class="{('dark' if st.session_state.dark_mode else '')}">
<div class="layout">
<div class="svg-col">
<svg viewBox="0 0 240 760" width="100%" style="display:block;">
<!-- 머리 -->
<ellipse cx="120" cy="56" rx="38" ry="44" fill="white" stroke="#CCC" stroke-width="1.2"/>
<path d="M82,46 Q75,46 73,55 Q71,64 73,72 Q75,80 82,80" fill="white" stroke="#CCC" stroke-width="1"/>
<path d="M158,46 Q165,46 167,55 Q169,64 167,72 Q165,80 158,80" fill="white" stroke="#CCC" stroke-width="1"/>
<path d="M104,48 Q110,44 117,46" fill="none" stroke="#C8B8A8" stroke-width="1" stroke-linecap="round"/>
<path d="M123,46 Q130,44 136,48" fill="none" stroke="#C8B8A8" stroke-width="1" stroke-linecap="round"/>
<ellipse cx="110" cy="58" rx="7" ry="5" fill="#F0F0EE" stroke="#DDD" stroke-width=".7"/>
<ellipse cx="130" cy="58" rx="7" ry="5" fill="#F0F0EE" stroke="#DDD" stroke-width=".7"/>
<circle cx="110" cy="58" r="3" fill="#7A6050"/><circle cx="130" cy="58" r="3" fill="#7A6050"/>
<circle cx="111" cy="57" r="1.1" fill="white"/><circle cx="131" cy="57" r="1.1" fill="white"/>
<path d="M117,70 Q120,77 123,70" fill="none" stroke="#C8A888" stroke-width="1" stroke-linecap="round"/>
<path d="M112,84 Q120,90 128,84" fill="none" stroke="#C0A090" stroke-width="1.2" stroke-linecap="round"/>
<!-- 목 -->
<rect x="111" y="96" width="18" height="30" rx="6" fill="white" stroke="#CCC" stroke-width="1"/>
<!-- 몸통 -->
<path d="M78,124 L94,119 L120,117 L146,119 L162,124 L165,154 L166,202 L166,308 L164,358 L160,376 L154,380 L154,402 L86,402 L80,380 L76,358 L74,308 L74,202 L75,154 Z" fill="white" stroke="#CCC" stroke-width="1.2"/>
<!-- 팔 -->
<path d="M78,126 Q64,131 57,148 L47,200 L45,262 L49,318 L57,330 L65,318 L67,264 L69,206 L77,158 L83,138 Z" fill="white" stroke="#CCC" stroke-width="1"/>
<ellipse cx="52" cy="344" rx="9" ry="13" fill="white" stroke="#CCC" stroke-width=".9"/>
<path d="M162,126 Q176,131 183,148 L193,200 L195,262 L191,318 L183,330 L175,318 L173,264 L171,206 L163,158 L157,138 Z" fill="white" stroke="#CCC" stroke-width="1"/>
<ellipse cx="188" cy="344" rx="9" ry="13" fill="white" stroke="#CCC" stroke-width=".9"/>
<!-- 오른 다리 -->
<path d="M86,402 L100,399 L111,399 L113,456 L112,516 L109,550 L93,553 L87,535 L86,476 Z" fill="white" stroke="#CCC" stroke-width="1"/>
<ellipse cx="99" cy="557" rx="13" ry="10" fill="white" stroke="#CCC" stroke-width="1"/>
<path d="M88,565 Q86,602 88,638 L91,668 L99,676 L107,668 L110,638 L112,602 L110,565 Z" fill="white" stroke="#CCC" stroke-width="1"/>
<path d="M90,672 Q87,683 84,692 Q82,699 90,702 L110,702 Q118,700 118,693 L114,683 L108,672 Z" fill="white" stroke="#CCC" stroke-width="1"/>
<!-- 왼 다리 -->
<path d="M154,402 L140,399 L129,399 L127,456 L128,516 L131,550 L147,553 L153,535 L154,476 Z" fill="white" stroke="#CCC" stroke-width="1"/>
<ellipse cx="141" cy="557" rx="13" ry="10" fill="white" stroke="#CCC" stroke-width="1"/>
<path d="M152,565 Q154,602 152,638 L149,668 L141,676 L133,668 L130,638 L128,602 L130,565 Z" fill="white" stroke="#CCC" stroke-width="1"/>
<path d="M150,672 Q153,683 156,692 Q158,699 150,702 L130,702 Q122,700 122,693 L126,683 L132,672 Z" fill="white" stroke="#CCC" stroke-width="1"/>
<!-- ══ 장기 ══ -->
<g class="ob" id="ob-brain" data-part="brain">
  <ellipse cx="120" cy="46" rx="26" ry="22" fill="#E8EEF8" stroke="#90A4CC" stroke-width="1.2"/>
  <path d="M96,42 Q104,31 120,29 Q136,31 144,42" fill="none" stroke="#90A4CC" stroke-width=".8" opacity=".6"/>
  <path d="M97,51 Q106,42 120,40 Q134,42 143,51" fill="none" stroke="#90A4CC" stroke-width=".6" opacity=".4"/>
  <line x1="120" y1="25" x2="120" y2="68" stroke="#90A4CC" stroke-width=".5" opacity=".3"/>
</g>
<g class="ob" id="ob-eye" data-part="eye">
  <ellipse cx="110" cy="58" rx="7" ry="5" fill="#CCE4F4" stroke="#60A0C8" stroke-width=".9" opacity=".8"/>
  <ellipse cx="130" cy="58" rx="7" ry="5" fill="#CCE4F4" stroke="#60A0C8" stroke-width=".9" opacity=".8"/>
</g>
<g class="ob" id="ob-ear" data-part="ear">
  <rect x="62" y="44" width="22" height="40" rx="8" fill="transparent" stroke="none"/>
  <rect x="156" y="44" width="22" height="40" rx="8" fill="transparent" stroke="none"/>
</g>
<g class="ob" id="ob-thyroid" data-part="thyroid">
  <path d="M112,114 Q107,108 107,115 Q107,124 115,127 L120,128 L125,127 Q133,124 133,115 Q133,108 128,114 Q125,118 120,119 Q115,118 112,114 Z" fill="#FCE8C4" stroke="#D0A040" stroke-width="1.1"/>
</g>
<g class="ob" id="ob-esophagus" data-part="esophagus">
  <rect x="117" y="124" width="6" height="44" rx="3" fill="#EEE098" stroke="#C0A840" stroke-width=".8"/>
</g>
<g class="ob" id="ob-spine" data-part="spine">
  <rect x="118" y="132" width="4" height="226" rx="2" fill="#EEEADC" stroke="#B8B090" stroke-width=".8"/>
  <rect x="116" y="139" width="8" height="5.5" rx="1.8" fill="#EEEADC" stroke="#B8B090" stroke-width=".6"/>
  <rect x="116" y="157" width="8" height="5.5" rx="1.8" fill="#EEEADC" stroke="#B8B090" stroke-width=".6"/>
  <rect x="116" y="175" width="8" height="5.5" rx="1.8" fill="#EEEADC" stroke="#B8B090" stroke-width=".6"/>
  <rect x="116" y="193" width="8" height="5.5" rx="1.8" fill="#EEEADC" stroke="#B8B090" stroke-width=".6"/>
  <rect x="116" y="211" width="8" height="5.5" rx="1.8" fill="#EEEADC" stroke="#B8B090" stroke-width=".6"/>
  <rect x="116" y="229" width="8" height="5.5" rx="1.8" fill="#EEEADC" stroke="#B8B090" stroke-width=".6"/>
  <rect x="116" y="247" width="8" height="5.5" rx="1.8" fill="#EEEADC" stroke="#B8B090" stroke-width=".6"/>
  <rect x="116" y="265" width="8" height="5.5" rx="1.8" fill="#EEEADC" stroke="#B8B090" stroke-width=".6"/>
  <rect x="116" y="283" width="8" height="5.5" rx="1.8" fill="#EEEADC" stroke="#B8B090" stroke-width=".6"/>
  <rect x="116" y="301" width="8" height="5.5" rx="1.8" fill="#EEEADC" stroke="#B8B090" stroke-width=".6"/>
  <rect x="116" y="319" width="8" height="5.5" rx="1.8" fill="#EEEADC" stroke="#B8B090" stroke-width=".6"/>
  <rect x="116" y="337" width="8" height="5.5" rx="1.8" fill="#EEEADC" stroke="#B8B090" stroke-width=".6"/>
</g>
<g class="ob" id="ob-lungs" data-part="lungs">
  <path d="M96,136 Q84,139 79,155 L75,202 Q73,229 83,241 Q93,250 107,246 L109,219 L110,136 Z" fill="#F4C4C4" stroke="#BC8080" stroke-width="1.3"/>
  <path d="M84,160 Q87,176 85,205" fill="none" stroke="#BC8080" stroke-width=".6" opacity=".5"/>
  <path d="M144,136 Q156,139 161,155 L165,202 Q167,229 157,241 Q147,250 133,246 L131,219 L130,136 Z" fill="#F4C4C4" stroke="#BC8080" stroke-width="1.3"/>
</g>
<g class="ob" id="ob-heart" data-part="heart">
  <path d="M120,160 Q106,150 96,161 Q85,172 96,186 L120,211 L144,186 Q155,172 144,161 Q134,150 120,160 Z" fill="#EEA0A0" stroke="#B85050" stroke-width="1.5"/>
  <path d="M108,167 Q103,176 108,186" fill="none" stroke="#E8BCBC" stroke-width=".9" opacity=".5"/>
</g>
<g class="ob" id="ob-liver" data-part="liver">
  <path d="M129,224 Q148,218 162,226 L165,251 Q161,273 146,274 Q132,274 123,265 Q116,256 122,228 Z" fill="#E8B880" stroke="#B07838" stroke-width="1.2"/>
</g>
<g class="ob" id="ob-gallbladder" data-part="gallbladder">
  <ellipse cx="155" cy="280" rx="8" ry="10" fill="#D4D880" stroke="#909030" stroke-width="1"/>
</g>
<g class="ob" id="ob-stomach" data-part="stomach">
  <path d="M91,228 Q77,231 72,247 Q67,264 79,276 Q90,284 109,282 L111,256 L103,228 Z" fill="#EEC880" stroke="#B88840" stroke-width="1.2"/>
</g>
<g class="ob" id="ob-spleen" data-part="spleen">
  <ellipse cx="71" cy="262" rx="11" ry="14" fill="#D0B4DC" stroke="#9060B0" stroke-width="1.1"/>
</g>
<g class="ob" id="ob-pancreas" data-part="pancreas">
  <path d="M90,290 Q111,284 142,288 L144,299 Q120,306 94,302 Z" fill="#E8D478" stroke="#A89828" stroke-width="1.1"/>
</g>
<g class="ob" id="ob-kidney" data-part="kidney">
  <path d="M77,304 Q67,304 65,315 Q63,329 72,336 Q80,341 88,335 L90,315 Q90,304 77,304 Z" fill="#DEAAB8" stroke="#A85870" stroke-width="1.2"/>
  <path d="M163,304 Q173,304 175,315 Q177,329 168,336 Q160,341 152,335 L150,315 Q150,304 163,304 Z" fill="#DEAAB8" stroke="#A85870" stroke-width="1.2"/>
</g>
<g class="ob" id="ob-intestine" data-part="intestine">
  <path d="M90,310 L89,342 Q89,362 107,364 L133,364 Q151,362 151,342 L150,310" fill="none" stroke="#E0B868" stroke-width="8" stroke-linecap="round" stroke-linejoin="round" opacity=".7"/>
  <path d="M97,324 Q105,315 113,324 Q121,333 129,324 Q137,315 143,324 Q149,333 147,342 Q140,350 133,343 Q126,336 120,343 Q114,350 107,343 Q100,334 97,324 Z" fill="#EED08C" stroke="#C0A040" stroke-width=".9"/>
</g>
<g class="ob" id="ob-bladder" data-part="bladder">
  <ellipse cx="120" cy="378" rx="17" ry="13" fill="#C0D4EE" stroke="#5888B8" stroke-width="1.2"/>
</g>
<g class="ob" id="ob-joints" data-part="joints">
  <circle cx="80"  cy="130" r="10" fill="#EEEADC" stroke="#A8A478" stroke-width="1.1" opacity=".9"/>
  <circle cx="160" cy="130" r="10" fill="#EEEADC" stroke="#A8A478" stroke-width="1.1" opacity=".9"/>
  <circle cx="99"  cy="555" r="10" fill="#EEEADC" stroke="#A8A478" stroke-width="1.1" opacity=".9"/>
  <circle cx="141" cy="555" r="10" fill="#EEEADC" stroke="#A8A478" stroke-width="1.1" opacity=".9"/>
</g>
<g class="ob" id="ob-legs" data-part="legs">
  <rect x="85" y="410" width="30" height="150" rx="10" fill="rgba(140,140,220,.12)" stroke="#8888C8" stroke-width="1.8" stroke-dasharray="5 3"/>
  <rect x="125" y="410" width="30" height="150" rx="10" fill="rgba(140,140,220,.12)" stroke="#8888C8" stroke-width="1.8" stroke-dasharray="5 3"/>
  <text x="100" y="490" text-anchor="middle" font-family="-apple-system,sans-serif" font-size="7.5" font-weight="600" fill="#404088">정맥</text>
  <text x="140" y="490" text-anchor="middle" font-family="-apple-system,sans-serif" font-size="7.5" font-weight="600" fill="#404088">정맥</text>
</g>
<!-- 라벨 (pointer-events 없음) -->
<g font-family="-apple-system,BlinkMacSystemFont,sans-serif" text-anchor="middle" font-size="8" font-weight="600" pointer-events="none">
  <text x="120" y="49"  fill="#4858A0">뇌</text>
  <text x="120" y="123" fill="#907030">갑상선</text>
  <text x="120" y="186" fill="#884040">심장</text>
  <text x="90"  y="197" fill="#884848">폐</text>
  <text x="150" y="197" fill="#884848">폐</text>
  <text x="146" y="254" fill="#7A5020">간</text>
  <text x="90"  y="257" fill="#7A5820">위</text>
  <text x="71"  y="265" fill="#603888" font-size="7.5">비장</text>
  <text x="120" y="298" fill="#706020">췌장</text>
  <text x="76"  y="323" fill="#784058">신장</text>
  <text x="164" y="323" fill="#784058">신장</text>
  <text x="120" y="344" fill="#7A5828">장</text>
  <text x="120" y="381" fill="#285880">방광</text>
  <text x="155" y="283" fill="#5A6820" font-size="7">담낭</text>
</g>
<!-- 범례 -->
<rect x="4" y="716" width="232" height="28" rx="7" fill="white" stroke="#E8E8E8" stroke-width=".8"/>
<circle cx="16"  cy="730" r="5" fill="#E8EEF8" stroke="#90A4CC" stroke-width=".8"/>
<text x="25"  y="734" font-size="8" fill="#999" font-family="-apple-system,sans-serif">비활성</text>
<circle cx="74"  cy="730" r="5" fill="#FFF0C0" stroke="#C8A030" stroke-width=".8"/>
<text x="83"  y="734" font-size="8" fill="#999" font-family="-apple-system,sans-serif">낮은 연관</text>
<circle cx="148" cy="730" r="5" fill="#FFB880" stroke="#C86030" stroke-width=".8"/>
<text x="157" y="734" font-size="8" fill="#999" font-family="-apple-system,sans-serif">중간</text>
<circle cx="194" cy="730" r="5" fill="#FF7868" stroke="#C03020" stroke-width=".8"/>
<text x="203" y="734" font-size="8" fill="#999" font-family="-apple-system,sans-serif">높음</text>
</svg>
<div class="hint">장기를 클릭하면 상세 정보가 표시됩니다</div>
</div>

<div class="panel-col">
  <div id="systemicSection" style="display:none;">
    <div style="font-size:10px;font-weight:600;color:#999;text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px;">전신계 (클릭하여 상세 보기)</div>
    <div class="systemic-grid" id="systemicGrid"></div>
  </div>
  <div class="card" id="defaultCard">
    <div class="card-name">인체 해부도</div>
    <div class="card-sub">장기를 클릭하면 상세 정보가 표시됩니다</div>
    <div class="desc">혈액·피부·림프·면역계는 위 버튼으로 확인하세요.</div>
    <div class="sec">연관 부위</div>
    <div class="tags" id="activeTags"><span class="tag tag-n">증상 선택 후 표시됩니다</span></div>
  </div>
  <div class="card" id="detailCard" style="display:none;">
    <div class="card-name" id="dName"></div>
    <div class="card-sub"  id="dSub"></div>
    <div class="bar-row">
      <div class="bar-lbl">연관도</div>
      <div class="bar-track"><div class="bar-fill" id="dBar" style="width:0%"></div></div>
      <div class="bar-pct"  id="dPct">0%</div>
    </div>
    <div class="sec">기능·병리</div>
    <div class="desc" id="dDesc"></div>
    <div class="sec">연관 질병</div>
    <div class="tags" id="dDiseases"></div>
    <button class="back-btn" onclick="showDefault();">← 목록으로</button>
  </div>
</div>
</div>

<script>
const PD={part_data_json};
const AP={active_json};
const SYSTEMIC={systemic_json};

function intColor(pct){{
  if(pct>=55) return {{f:'#FF8068',s:'#C03020'}};
  if(pct>=28) return {{f:'#FFB878',s:'#C86028'}};
  if(pct> 0)  return {{f:'#FFF0B8',s:'#C8A028'}};
  return null;
}}
function barGrad(pct){{
  if(pct>=55) return 'linear-gradient(90deg,#f6ad55,#fc8181)';
  if(pct>=28) return 'linear-gradient(90deg,#68d391,#f6ad55)';
  return '#68d391';
}}
function sysStyle(pct){{
  if(pct>=55) return {{dot:'#FF8068',border:'#E03020',text:'#A32D2D',bg:'#FCEBEB'}};
  if(pct>=28) return {{dot:'#FFB878',border:'#C86028',text:'#854F0B',bg:'#FAEEDA'}};
  if(pct> 0)  return {{dot:'#d4edda',border:'#C8A028',text:'#2e7d32',bg:'#EAF3DE'}};
  return {{dot:'#DDD',border:'#CCC',text:'#888',bg:'#F5F5F3'}};
}}

Object.keys(AP).forEach(part=>{{
  if(SYSTEMIC.includes(part)) return;
  const pct=Math.round(AP[part]*100);
  const zone=document.getElementById('ob-'+part);
  if(!zone) return;
  const c=intColor(pct); if(!c) return;
  zone.querySelectorAll('path,ellipse,rect,circle').forEach(el=>{{
    const f=el.getAttribute('fill');
    if(f&&f!=='none'&&!f.startsWith('rgba')&&!f.startsWith('transparent')){{
      el.setAttribute('fill',c.f); el.setAttribute('stroke',c.s);
    }}
  }});
}});

const sorted=Object.keys(AP).sort((a,b)=>AP[b]-AP[a]);
const atEl=document.getElementById('activeTags');
atEl.innerHTML = sorted.length ? sorted.slice(0,12).map(p=>{{
  const pct=Math.round(AP[p]*100);
  const cls=pct>=55?'tag-r':pct>=28?'tag-o':'tag-g';
  const nm=PD[p]?PD[p].name:p;
  return `<span class="tag ${{cls}}" style="cursor:pointer" onclick="showDetail('${{p}}')">${{nm}} ${{pct}}%</span>`;
}}).join('') : '<span class="tag tag-n">증상 선택 후 표시됩니다</span>';

const systemicParts=SYSTEMIC.filter(p=>PD[p]);
const secEl=document.getElementById('systemicSection');
const grid=document.getElementById('systemicGrid');
if(systemicParts.length>0){{
  secEl.style.display='block';
  systemicParts.forEach(part=>{{
    const d=PD[part]; const pct=d.intensity||0; const st=sysStyle(pct);
    const btn=document.createElement('button');
    btn.className='sys-btn';
    btn.style.cssText=`border-color:${{st.border}};background:${{st.bg}};`;
    btn.innerHTML=`<span class="sys-dot" style="background:${{st.dot}};border:1.5px solid ${{st.border}};"></span>`+
      `<span class="sys-name" style="color:${{st.text}};">${{d.name}}</span>`+
      `<span class="sys-pct" style="color:${{st.text}};">${{pct}}%</span>`;
    btn.onclick=()=>showDetail(part); grid.appendChild(btn);
  }});
}}

function showDetail(part){{
  const d=PD[part]; if(!d) return;
  document.getElementById('defaultCard').style.display='none';
  document.getElementById('detailCard').style.display='block';
  document.getElementById('dName').textContent=d.name;
  document.getElementById('dSub').textContent=d.sub;
  document.getElementById('dDesc').textContent=d.desc;
  const pct=d.intensity||0;
  const bar=document.getElementById('dBar');
  bar.style.width=pct+'%'; bar.style.background=barGrad(pct);
  document.getElementById('dPct').textContent=pct+'%';
  const dd=document.getElementById('dDiseases'); dd.innerHTML='';
  if(d.diseases&&d.diseases.length>0){{
    d.diseases.forEach(x=>{{
      const cls=x.prob>=30?'tag-r':x.prob>=15?'tag-o':'tag-g';
      dd.innerHTML+=`<span class="tag ${{cls}}">${{x.kr}} ${{x.prob.toFixed(1)}}%</span>`;
    }});
  }} else {{ dd.innerHTML='<span class="tag tag-n">예측된 질병 없음</span>'; }}
}}
function showDefault(){{
  document.getElementById('defaultCard').style.display='block';
  document.getElementById('detailCard').style.display='none';
}}
document.querySelectorAll('.ob').forEach(el=>{{
  el.addEventListener('click',()=>showDetail(el.getAttribute('data-part')));
}});
if(sorted.length>0) showDetail(sorted[0]);
</script>
</body></html>"""
    components.html(html, height=800, scrolling=False)

# ════════════════════════════════════════════════════════
#  성별·연령대 가중치
# ════════════════════════════════════════════════════════
AGE_GENDER_WEIGHTS = {
    "Heart attack":       {("남","10대"):0.2,("남","20대"):0.4,("남","30대"):0.7,("남","40대"):1.2,("남","50대"):1.8,("남","60대+"):2.5,("여","10대"):0.1,("여","20대"):0.2,("여","30대"):0.4,("여","40대"):0.8,("여","50대"):1.4,("여","60대+"):2.0},
    "Hypertension":       {("남","10대"):0.3,("남","20대"):0.6,("남","30대"):0.9,("남","40대"):1.3,("남","50대"):1.8,("남","60대+"):2.2,("여","10대"):0.2,("여","20대"):0.4,("여","30대"):0.7,("여","40대"):1.0,("여","50대"):1.6,("여","60대+"):2.0},
    "Varicose veins":     {("남","10대"):0.3,("남","20대"):0.5,("남","30대"):0.7,("남","40대"):1.0,("남","50대"):1.3,("남","60대+"):1.5,("여","10대"):0.5,("여","20대"):1.2,("여","30대"):1.5,("여","40대"):1.6,("여","50대"):1.4,("여","60대+"):1.3},
    "Diabetes":           {("남","10대"):0.4,("남","20대"):0.6,("남","30대"):0.9,("남","40대"):1.3,("남","50대"):1.7,("남","60대+"):2.0,("여","10대"):0.4,("여","20대"):0.6,("여","30대"):0.9,("여","40대"):1.2,("여","50대"):1.5,("여","60대+"):1.8},
    "Hypothyroidism":     {("남","10대"):0.3,("남","20대"):0.3,("남","30대"):0.4,("남","40대"):0.5,("남","50대"):0.6,("남","60대+"):0.7,("여","10대"):0.8,("여","20대"):1.5,("여","30대"):1.8,("여","40대"):2.0,("여","50대"):2.2,("여","60대+"):2.0},
    "Hyperthyroidism":    {("남","10대"):0.3,("남","20대"):0.4,("남","30대"):0.4,("남","40대"):0.5,("남","50대"):0.5,("남","60대+"):0.5,("여","10대"):0.8,("여","20대"):1.6,("여","30대"):1.8,("여","40대"):1.7,("여","50대"):1.5,("여","60대+"):1.2},
    "Osteoarthritis":     {("남","10대"):0.1,("남","20대"):0.2,("남","30대"):0.4,("남","40대"):0.8,("남","50대"):1.5,("남","60대+"):2.2,("여","10대"):0.1,("여","20대"):0.2,("여","30대"):0.4,("여","40대"):1.0,("여","50대"):1.8,("여","60대+"):2.5},
    "Arthritis":          {("남","10대"):0.3,("남","20대"):0.5,("남","30대"):0.7,("남","40대"):1.0,("남","50대"):1.4,("남","60대+"):1.8,("여","10대"):0.5,("여","20대"):0.8,("여","30대"):1.2,("여","40대"):1.5,("여","50대"):1.8,("여","60대+"):2.0},
    "Acne":               {("남","10대"):2.2,("남","20대"):1.5,("남","30대"):0.7,("남","40대"):0.4,("남","50대"):0.2,("남","60대+"):0.1,("여","10대"):2.0,("여","20대"):1.4,("여","30대"):0.8,("여","40대"):0.5,("여","50대"):0.3,("여","60대+"):0.1},
    "Migraine":           {("남","10대"):0.8,("남","20대"):0.9,("남","30대"):0.8,("남","40대"):0.7,("남","50대"):0.5,("남","60대+"):0.4,("여","10대"):1.2,("여","20대"):1.8,("여","30대"):2.0,("여","40대"):1.8,("여","50대"):1.3,("여","60대+"):0.8},
    "Urinary tract infection":{("남","10대"):0.3,("남","20대"):0.3,("남","30대"):0.4,("남","40대"):0.5,("남","50대"):0.8,("남","60대+"):1.2,("여","10대"):1.2,("여","20대"):1.8,("여","30대"):1.8,("여","40대"):1.6,("여","50대"):1.4,("여","60대+"):1.3},
    "Bronchial Asthma":   {("남","10대"):1.8,("남","20대"):1.3,("남","30대"):1.0,("남","40대"):0.9,("남","50대"):0.8,("남","60대+"):0.9,("여","10대"):1.4,("여","20대"):1.3,("여","30대"):1.2,("여","40대"):1.1,("여","50대"):1.0,("여","60대+"):0.9},
}

def apply_age_gender_weight(result_rows, gender, age_group):
    if gender == "선택 안 함":
        return result_rows
    weighted = []
    for r in result_rows:
        w = AGE_GENDER_WEIGHTS.get(r["disease"], {}).get((gender, age_group), 1.0)
        weighted.append({**r, "probability": r["probability"] * w})
    total = sum(r["probability"] for r in weighted)
    if total > 0:
        weighted = [{**r, "probability": r["probability"] / total} for r in weighted]
    return weighted

# ════════════════════════════════════════════════════════
#  사이드바
# ════════════════════════════════════════════════════════
if "reset_trigger" not in st.session_state:
    st.session_state.reset_trigger = 0
if "ai_comment" not in st.session_state:
    st.session_state.ai_comment = ""
if "ai_requested" not in st.session_state:
    st.session_state.ai_requested = False

categories = {
    "전신 증상": ["fatigue","weight_loss","weight_gain","lethargy","malaise","restlessness","anxiety","mood_swings"],
    "발열·통증":  ["high_fever","mild_fever","headache","joint_pain","back_pain","stomach_pain","chest_pain","abdominal_pain","pain_behind_the_eyes"],
    "피부·외형":  ["itching","skin_rash","nodal_skin_eruptions","yellowish_skin","yellowing_of_eyes","dark_urine"],
    "소화기":     ["vomiting","nausea","indigestion","acidity","diarrhoea","constipation","loss_of_appetite"],
    "호흡기":     ["cough","breathlessness","phlegm","congestion","runny_nose","sinus_pressure","throat_irritation","continuous_sneezing"],
    "기타":       ["sweating","chills","shivering","dehydration","blurred_and_distorted_vision","fast_heart_rate","burning_micturition","weakness_in_limbs"],
}
ALL_SYMS_FOR_RESET = [s for syms in categories.values() for s in syms]

with st.sidebar:
    # ── 다크모드 토글 (최상단)
    dm_col1, dm_col2 = st.columns([3,1])
    with dm_col1:
        st.markdown("## 🏥 증상 분석기")
    with dm_col2:
        dm_label = "☀️" if st.session_state.dark_mode else "🌙"
        if st.button(dm_label, key="dark_toggle", help="다크/라이트 모드 전환"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    st.caption("증상을 선택하면 AI가 질병을 예측합니다")
    st.divider()

    st.markdown("#### 👤 기본 정보")
    st.caption("선택 시 인구통계별 가중치가 적용됩니다")
    col_g, col_a = st.columns(2)
    with col_g:
        gender = st.selectbox("성별", ["선택 안 함","남","여"], key="gender_select",
                              label_visibility="collapsed")
        st.caption("🧑 성별")
    with col_a:
        age_group = st.selectbox("연령대", ["10대","20대","30대","40대","50대","60대+"],
                                 index=2, key="age_select", label_visibility="collapsed")
        st.caption("📅 연령대")
    if gender != "선택 안 함":
        st.success(f"✅ {gender}성 · {age_group} 가중치 적용 중")
    st.divider()

    st.markdown("#### 🔍 증상 검색")
    all_sym_options = {SYMPTOM_KR[s]: s for s in ALL_SYMS_FOR_RESET if s in SYMPTOM_KR}
    search_selected_kr = st.multiselect(
        "증상 검색", options=list(all_sym_options.keys()),
        placeholder="증상 이름을 입력하거나 선택하세요",
        label_visibility="collapsed",
        key=f"sym_search_{st.session_state.reset_trigger}",
    )
    search_selected = [all_sym_options[kr] for kr in search_selected_kr]
    st.divider()

    st.markdown("#### ☑️ 카테고리별 선택")
    checkbox_selected = []
    for cat, syms in categories.items():
        with st.expander(cat, expanded=False):
            for s in syms:
                if s in SYMPTOM_KR:
                    if st.checkbox(SYMPTOM_KR[s],
                                   key=f"cb_{s}_{st.session_state.reset_trigger}",
                                   value=(s in search_selected)):
                        checkbox_selected.append(s)

    selected_symptoms = list(dict.fromkeys(search_selected + checkbox_selected))
    st.divider()

    if selected_symptoms:
        badge_str = "".join([
            f"<span style='display:inline-block;background:#E6F1FB;color:#185FA5;"
            f"border:1px solid #B5D4F4;border-radius:999px;font-size:11px;"
            f"padding:2px 9px;margin:2px;'>{SYMPTOM_KR.get(s,s)}</span>"
            for s in selected_symptoms
        ])
        st.markdown(
            f"<div style='line-height:2;'><span style='font-size:12px;font-weight:600;"
            f"color:#555;'>선택된 증상 ({len(selected_symptoms)}개)</span><br>{badge_str}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("")
        if st.button("🔄 전체 초기화", use_container_width=True, type="secondary"):
            st.session_state.reset_trigger += 1
            st.session_state.ai_comment = ""
            st.session_state.ai_requested = False
            st.rerun()
    st.divider()

    top_n        = st.slider("상위 N개 질병", 3, 15, 8)
    model_choice = st.radio("예측 모델", ["앙상블 (권장)","Naive Bayes","Random Forest"])
    st.divider()
    st.markdown(
        "<div style='font-size:11px;color:#888;line-height:1.7;'>"
        "<b>데이터 출처</b><br>"
        "· <a href='https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning' target='_blank'>Kaggle kaushil268</a><br>"
        "· <a href='https://people.dbmi.columbia.edu/~friedma/Projects/DiseaseSymptomKB/index.html' target='_blank'>Columbia DBMI KB</a><br>"
        "· <a href='https://hpo.jax.org/data/annotations' target='_blank'>HPO JAX.org</a>"
        "</div>", unsafe_allow_html=True,
    )

# ════════════════════════════════════════════════════════
#  메인
# ════════════════════════════════════════════════════════
st.title("🩺 증상 기반 질병·인체 시각화 대시보드")

hcol1, hcol2 = st.columns([4,1])
with hcol1:
    st.caption("ML 확률 예측 · SVG 인체 해부도 · 약품·치료법·민간요법 안내")
with hcol2:
    if gender != "선택 안 함":
        st.markdown(
            f"<div style='text-align:right;font-size:12px;color:#185FA5;"
            f"background:#E6F1FB;padding:4px 10px;border-radius:8px;"
            f"border:1px solid #B5D4F4;'>👤 {gender}성 · {age_group}</div>",
            unsafe_allow_html=True,
        )

st.warning("⚠️ **면책 조항**: 이 도구는 의료기기가 아니며 진단·처방을 대체하지 않습니다. 약품 복용 전 반드시 의사·약사와 상담하세요.")

nb_model, rf_model, all_syms = train_models()

if not selected_symptoms:
    st.info("👈 왼쪽 사이드바에서 증상을 검색하거나 선택하면 예측 결과와 인체 해부도가 나타납니다.")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("지원 질병","41종"); c2.metric("분석 증상","132개")
    c3.metric("ML 모델","NB+RF 앙상블"); c4.metric("해부도 부위","22개")
    st.stop()

# 예측
input_vec = np.array([[1 if s in selected_symptoms else 0 for s in all_syms]])
nb_p = dict(zip(nb_model.classes_, nb_model.predict_proba(input_vec)[0]))
rf_p = dict(zip(rf_model.classes_, rf_model.predict_proba(input_vec)[0]))
result_rows = []
for d in DISEASE_SYMPTOMS:
    nb_v, rf_v = nb_p.get(d,0), rf_p.get(d,0)
    prob = (nb_v+rf_v)/2 if model_choice=="앙상블 (권장)" else (nb_v if model_choice=="Naive Bayes" else rf_v)
    result_rows.append({"disease":d,"disease_kr":DISEASE_KR.get(d,d),"probability":prob})

result_rows = apply_age_gender_weight(result_rows, gender, age_group)
result_df = (pd.DataFrame(result_rows).sort_values("probability",ascending=False)
             .head(top_n).reset_index(drop=True))
result_df["prob_pct"] = (result_df["probability"]*100).round(1)

# 신체 부위 활성화
part_intensity: dict = {}
part_disease_map: dict = {}
for _, row in result_df.iterrows():
    d, prob = row["disease"], row["probability"]
    for part in DISEASE_BODY_PARTS.get(d,[]):
        part_intensity[part] = min(1.0, part_intensity.get(part,0)+prob*2)
        part_disease_map.setdefault(part,[]).append(
            {"name":d,"kr":row["disease_kr"],"prob":row["prob_pct"]})
if part_intensity:
    mx = max(part_intensity.values())
    if mx>0: part_intensity = {p:v/mx for p,v in part_intensity.items()}

# ── 탭
tab1, tab2, tab3, tab4 = st.tabs(["📊 예측 결과", "🫀 인체 해부도", "💊 치료법 안내", "📄 PDF 리포트"])

# ════════════════════════════
with tab1:
    badge_html_tab = "".join([
        f"<span style='display:inline-block;background:#E6F1FB;color:#0C447C;"
        f"border:1px solid #B5D4F4;border-radius:999px;font-size:11px;"
        f"padding:2px 9px;margin:2px;'>{SYMPTOM_KR.get(s,s)}</span>"
        for s in selected_symptoms
    ])
    demo_badge = (
        f"<span style='display:inline-block;background:#EAF3DE;color:#27500A;"
        f"border:1px solid #97C459;border-radius:999px;font-size:11px;"
        f"padding:2px 9px;margin:2px;'>👤 {gender}성 · {age_group} 가중치 적용</span>"
    ) if gender != "선택 안 함" else ""
    st.markdown(
        f"<div style='margin-bottom:12px;line-height:2.2;'>"
        f"<span style='font-size:12px;font-weight:600;color:#555;'>선택 증상 {len(selected_symptoms)}개</span>　"
        f"{badge_html_tab} {demo_badge}</div>",
        unsafe_allow_html=True,
    )

    # ── AI 한줄 평가 박스
    st.markdown("---")
    ai_col1, ai_col2 = st.columns([5,1])
    with ai_col1:
        st.markdown("#### 🤖 AI 한줄 평가")
        st.caption("Anthropic Claude가 증상과 예측 질병을 분석해 한줄 코멘트를 제공합니다.")
    with ai_col2:
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        if st.button("✨ 평가 받기", type="primary", use_container_width=True):
            st.session_state.ai_requested = True
            top_d = [{"kr": row["disease_kr"], "pct": row["prob_pct"]}
                     for _, row in result_df.head(3).iterrows()]
            sym_kr = [SYMPTOM_KR.get(s,s) for s in selected_symptoms]
            with st.spinner("AI 분석 중..."):
                st.session_state.ai_comment = get_ai_comment(sym_kr, top_d, gender, age_group)

    if st.session_state.ai_comment:
        comment = st.session_state.ai_comment
        if comment.startswith("⚠️") or comment.startswith("API") or comment.startswith("연결"):
            st.warning(comment)
        else:
            st.markdown(
                f"<div style='background:linear-gradient(135deg,#e8f5e9,#f1f8e9);"
                f"border:1.5px solid #81c784;border-radius:12px;padding:16px 20px;"
                f"margin:8px 0 16px 0;'>"
                f"<div style='font-size:13px;color:#1b5e20;line-height:1.8;'>"
                f"🩺 {comment}</div></div>",
                unsafe_allow_html=True,
            )
    elif not st.session_state.ai_requested:
        st.markdown(
            "<div style='background:#f8f9fa;border:1px dashed #ccc;border-radius:10px;"
            "padding:14px 18px;color:#aaa;font-size:12px;margin-bottom:12px;'>"
            "오른쪽 버튼을 클릭하면 AI가 증상을 분석해 한줄 코멘트를 제공합니다.</div>",
            unsafe_allow_html=True,
        )
    st.markdown("---")

    col_chart, col_cards = st.columns([3,2])
    with col_chart:
        colors_bar = ["#E24B4A" if p>=30 else "#BA7517" if p>=15 else "#378ADD"
                      for p in result_df["prob_pct"]]
        fig = go.Figure(go.Bar(
            x=result_df["prob_pct"], y=result_df["disease_kr"], orientation="h",
            marker_color=colors_bar,
            text=[f"{p:.1f}%" for p in result_df["prob_pct"]], textposition="outside",
            hovertemplate="<b>%{y}</b><br>확률: %{x:.1f}%<extra></extra>",
        ))
        fig.update_layout(
            height=max(300,top_n*44), margin=dict(l=10,r=60,t=10,b=10),
            xaxis_title="가능성 (%)", yaxis=dict(autorange="reversed"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(size=13),
            xaxis=dict(range=[0, min(100, result_df["prob_pct"].max()*1.35)])
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "<span style='color:#E24B4A'>■</span> 높은 가능성 (≥30%)　"
            "<span style='color:#BA7517'>■</span> 중간 (15~30%)　"
            "<span style='color:#378ADD'>■</span> 낮음 (<15%)",
            unsafe_allow_html=True,
        )

    with col_cards:
        st.markdown("##### 상위 3개 질병")
        URG_STYLE = {
            "즉시 병원": ("🚨","#FCEBEB","#E24B4A","#A32D2D"),
            "빠른 진료": ("⚠️","#FAEEDA","#BA7517","#854F0B"),
            "경과 관찰": ("✅","#EAF3DE","#3B6D11","#27500A"),
        }
        for _, row in result_df.head(3).iterrows():
            p = row["prob_pct"]
            urg = TREATMENT_DB.get(row["disease"],{}).get("urgency","경과 관찰")
            icon,bg,bd,tc = URG_STYLE.get(urg,("ℹ️","#E6F1FB","#185FA5","#0C447C"))
            parts_kr = [BODY_PART_KR.get(pt,"") for pt in DISEASE_BODY_PARTS.get(row["disease"],[])[:3]]
            naver_url = f"https://map.naver.com/v5/search/{row['disease_kr'].replace(' ','+')}+병원"
            hospital_btn = ""
            if urg in ("즉시 병원","빠른 진료"):
                hospital_btn = (
                    f"<div style='margin-top:8px;'>"
                    f"<a href='{naver_url}' target='_blank' style='display:block;text-align:center;"
                    f"background:white;color:{tc};border:1px solid {bd};border-radius:6px;"
                    f"padding:5px 0;font-size:12px;font-weight:600;text-decoration:none;'>"
                    f"🗺 네이버지도에서 근처 병원 찾기</a></div>"
                )
            st.markdown(
                f"<div style='background:{bg};border:1.5px solid {bd};border-radius:10px;"
                f"padding:.8rem 1rem;margin-bottom:10px;'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;'>"
                f"<span style='font-size:12px;font-weight:600;color:{tc};background:white;"
                f"padding:2px 8px;border-radius:999px;border:1px solid {bd};'>{icon} {urg}</span>"
                f"<span style='font-size:11px;color:{tc};opacity:.6;'>{row['disease']}</span></div>"
                f"<div style='font-size:16px;font-weight:500;color:{tc};margin-bottom:2px;'>{row['disease_kr']}</div>"
                f"<div style='display:flex;justify-content:space-between;align-items:flex-end;'>"
                f"<span style='font-size:26px;font-weight:700;color:{tc};'>{p:.1f}%</span>"
                f"<span style='font-size:11px;color:{tc};opacity:.65;text-align:right;'>{' · '.join(parts_kr)}</span>"
                f"</div>{hospital_btn}</div>",
                unsafe_allow_html=True,
            )

    with st.expander("📋 전체 결과 테이블"):
        disp = result_df[["disease_kr","disease","prob_pct"]].copy()
        disp.columns = ["질병명(한국어)","질병명(영어)","가능성(%)"]
        disp.index = range(1, len(disp)+1)
        st.dataframe(disp, use_container_width=True)

# ════════════════════════════
with tab2:
    st.markdown("##### 🫀 인체 해부도 — 예측 질병 연관 부위 시각화")
    st.caption("빨간색이 진할수록 연관도 높음 · 장기 클릭 또는 전신계 버튼으로 상세 정보 확인")
    sorted_parts = sorted(part_intensity.items(), key=lambda x: -x[1])
    badge_html = "".join([
        f"<span style='{'background:#FCEBEB;color:#A32D2D;border:1px solid #E24B4A;' if int(v*100)>=60 else 'background:#FAEEDA;color:#854F0B;border:1px solid #BA7517;' if int(v*100)>=30 else 'background:#EAF3DE;color:#27500A;border:1px solid #3B6D11;'}"
        f"border-radius:999px;font-size:12px;font-weight:500;padding:3px 11px;margin:2px;display:inline-block;'>"
        f"{BODY_PART_KR.get(p,p)} {int(v*100)}%</span>"
        for p,v in sorted_parts[:10]
    ])
    if badge_html:
        st.markdown(f"**주요 연관 부위:** {badge_html}", unsafe_allow_html=True)
        st.markdown("")
    render_body_anatomy(part_intensity, part_disease_map)

# ════════════════════════════
with tab3:
    st.markdown("##### 💊 질병별 약품·치료법·민간요법")
    st.caption("⚠️ 약품 복용 전 반드시 의사·약사와 상담하세요.")
    top_diseases = result_df["disease"].tolist()
    tab_labels = [
        f"{DISEASE_KR.get(d,d)} ({result_df.loc[result_df['disease']==d,'prob_pct'].values[0]:.1f}%)"
        for d in top_diseases
    ]
    d_tabs = st.tabs(tab_labels)
    for dtab, disease in zip(d_tabs, top_diseases):
        with dtab:
            info = TREATMENT_DB.get(disease)
            if not info:
                st.info("치료 정보 DB 추가 예정"); continue
            urg = info["urgency"]
            icon,bg,bd,tc = {
                "즉시 병원":("🚨","#FCEBEB","#E24B4A","#A32D2D"),
                "빠른 진료":("⚠️","#FAEEDA","#BA7517","#854F0B"),
                "경과 관찰":("✅","#EAF3DE","#3B6D11","#27500A"),
            }.get(urg,("ℹ️","#E6F1FB","#185FA5","#0C447C"))
            st.markdown(
                f"<div style='display:inline-block;background:{bg};border:1px solid {bd};"
                f"border-radius:8px;padding:4px 16px;font-size:13px;font-weight:500;"
                f"color:{tc};margin-bottom:8px;'>{icon} {urg}</div>",
                unsafe_allow_html=True,
            )
            # ── 질병 설명 추가
            desc = DISEASE_DESC.get(disease,"")
            if desc:
                st.markdown(
                    f"<div style='background:#f8f9fa;border-left:3px solid {bd};"
                    f"border-radius:0 8px 8px 0;padding:10px 14px;margin-bottom:14px;"
                    f"font-size:12.5px;color:#444;line-height:1.75;'>"
                    f"📖 {desc}</div>",
                    unsafe_allow_html=True,
                )
            c1,c2,c3 = st.columns(3)
            with c1:
                st.markdown("### 💊 추천 약품")
                for drug in info["drugs"]:
                    st.markdown(
                        f"<div style='border:0.5px solid #d0d0d0;border-radius:8px;"
                        f"padding:.6rem .9rem;margin-bottom:8px;'>"
                        f"<div style='font-size:14px;font-weight:500;'>{drug['name']}</div>"
                        f"<span style='background:#E6F1FB;color:#185FA5;padding:1px 7px;"
                        f"border-radius:4px;font-size:11px;'>{drug['type']}</span>"
                        f"<div style='font-size:12px;color:#666;margin-top:4px;'>{drug['note']}</div>"
                        f"</div>", unsafe_allow_html=True,
                    )
            with c2:
                st.markdown("### 🏥 치료·관리법")
                for i,t in enumerate(info["treatments"],1):
                    st.markdown(
                        f"<div style='display:flex;gap:8px;align-items:flex-start;margin-bottom:7px;'>"
                        f"<span style='background:#E1F5EE;color:#0F6E56;border-radius:50%;"
                        f"width:22px;height:22px;display:flex;align-items:center;justify-content:center;"
                        f"font-size:11px;font-weight:500;flex-shrink:0;'>{i}</span>"
                        f"<span style='font-size:13px;line-height:1.5;'>{t}</span></div>",
                        unsafe_allow_html=True,
                    )
            with c3:
                st.markdown("### 🌿 민간요법")
                st.caption("근거 수준이 다양합니다. 보조 수단으로만 활용하세요.")
                for remedy in info["folk_remedies"]:
                    st.markdown(
                        f"<div style='display:flex;gap:8px;align-items:flex-start;margin-bottom:7px;'>"
                        f"<span style='color:#3B6D11;font-size:14px;flex-shrink:0;'>🌱</span>"
                        f"<span style='font-size:13px;line-height:1.5;'>{remedy}</span></div>",
                        unsafe_allow_html=True,
                    )

# ════════════════════════════
with tab4:
    st.markdown("##### 📄 PDF 리포트 다운로드")
    st.caption("선택한 증상, 예측 질병, AI 평가, 치료법 정보를 한 장의 PDF로 저장합니다.")

    if not FONTS_OK:
        st.error("❌ 한국어 폰트 준비에 실패했습니다. 서버 환경을 확인해주세요.")
    else:
        pdf_col1, pdf_col2 = st.columns([3,2])
        with pdf_col1:
            st.markdown(
                "<div style='background:#f0f4ff;border:1px solid #c5d0f0;border-radius:10px;"
                "padding:16px 20px;'>"
                "<div style='font-size:14px;font-weight:600;color:#1e3a5f;margin-bottom:8px;'>📋 포함 내용</div>"
                "<ul style='font-size:12.5px;color:#444;line-height:2;margin:0;padding-left:18px;'>"
                "<li>선택 증상 목록</li>"
                "<li>AI 한줄 평가 (생성된 경우)</li>"
                "<li>질병 예측 결과 테이블</li>"
                "<li>주요 연관 신체 부위</li>"
                "<li>상위 5개 질병 상세 (설명·약품·치료법)</li>"
                "</ul></div>",
                unsafe_allow_html=True,
            )
        with pdf_col2:
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            if st.button("📥 PDF 생성 및 다운로드", type="primary", use_container_width=True):
                with st.spinner("PDF 생성 중... (처음엔 폰트 로딩으로 약 5~10초 소요)"):
                    try:
                        pdf_bytes = generate_pdf_report(
                            selected_symptoms, result_df, part_intensity,
                            gender, age_group, model_choice,
                            ai_comment=st.session_state.ai_comment,
                        )
                        fname = f"질병예측리포트_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                        st.download_button(
                            label="⬇️ PDF 저장하기",
                            data=pdf_bytes,
                            file_name=fname,
                            mime="application/pdf",
                            use_container_width=True,
                        )
                        st.success(f"✅ PDF 생성 완료! ({len(pdf_bytes)//1024} KB)")
                    except Exception as e:
                        st.error(f"PDF 생성 실패: {e}")

        st.markdown("---")
        st.markdown(
            "<div style='font-size:11px;color:#888;'>"
            "⚠️ 생성된 PDF는 참고용이며 의료 진단을 대체하지 않습니다.<br>"
            "AI 평가를 포함하려면 먼저 📊 예측 결과 탭에서 '✨ 평가 받기' 버튼을 눌러주세요."
            "</div>",
            unsafe_allow_html=True,
        )

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
**PDF**: ReportLab + NotoSansCJK-KR 폰트  
**AI 평가**: Claude Haiku (Anthropic API)  
**한계**: 공개 데이터 기반으로 실제 임상 정확도와 차이가 있습니다. 반드시 의사 진료를 받으세요.
    """)
