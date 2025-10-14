# recommendation_data.py
# 각 카테고리별 추천 데이터

BOOK_DATA = {
    "행복": [
        {"title": "해리 포터 시리즈", "author": "J.K. 롤링", "description": "마법 세계의 모험과 우정을 담은 판타지 소설"},
        {"title": "어린 왕자", "author": "생텍쥐페리", "description": "순수한 마음과 사랑에 대한 철학적 동화"},
        {"title": "데일 카네기 인간관계론", "author": "데일 카네기", "description": "긍정적인 인간관계 구축을 위한 실용적인 조언"},
        {"title": "행복의 기원", "author": "서은국", "description": "과학적 관점에서 바라본 행복의 본질"},
    ],
    "슬픔": [
        {"title": "상실의 시대", "author": "무라카미 하루키", "description": "청춘의 상실과 그리움을 섬세하게 그린 소설"},
        {"title": "달과 6펜스", "author": "서머싯 몸", "description": "예술에 대한 열정과 인생의 선택에 관한 이야기"},
        {"title": "위로", "author": "정여울", "description": "상처받은 마음을 다독이는 따뜻한 에세이"},
        {"title": "언어의 온도", "author": "이기주", "description": "따뜻한 말들의 힘을 느낄 수 있는 책"},
    ],
    "분노": [
        {"title": "정의란 무엇인가", "author": "마이클 샌델", "description": "정의와 윤리에 대한 철학적 성찰"},
        {"title": "82년생 김지영", "author": "조남주", "description": "현대 사회의 불합리함을 다룬 공감의 소설"},
        {"title": "총, 균, 쇠", "author": "재레드 다이아몬드", "description": "인류 문명의 불평등에 대한 통찰"},
        {"title": "분노와 슬픔", "author": "김형경", "description": "감정을 이해하고 다스리는 방법"},
    ],
    "평온": [
        {"title": "월든", "author": "헨리 데이빗 소로", "description": "자연 속에서의 단순한 삶에 대한 명상"},
        {"title": "느리게 사는 즐거움", "author": "칼 오노레", "description": "천천히 살아가는 삶의 가치"},
        {"title": "리틀 포레스트", "author": "이가라시 다이스케", "description": "시골 생활의 소소한 행복을 담은 만화"},
        {"title": "나는 나로 살기로 했다", "author": "김수현", "description": "자기 자신으로 사는 평온함"},
    ],
    "불안": [
        {"title": "불안", "author": "알랭 드 보통", "description": "현대인의 불안을 철학적으로 분석"},
        {"title": "아몬드", "author": "손원평", "description": "감정을 이해하지 못하는 소년의 성장 이야기"},
        {"title": "나미야 잡화점의 기적", "author": "히가시노 게이고", "description": "고민에 대한 따뜻한 답변이 담긴 소설"},
        {"title": "어쩌면 별들이 너의 슬픔을 가져갈지도 몰라", "author": "김용택", "description": "불안한 마음을 위로하는 시집"},
    ]
}

MUSIC_DATA = {
    "행복": [
        {"title": "Happy", "artist": "Pharrell Williams", "genre": "팝", "description": "밝고 경쾌한 리듬의 행복 찬가"},
        {"title": "Blueming", "artist": "아이유", "description": "사랑스럽고 달콤한 감성의 곡"},
        {"title": "Good Day", "artist": "아이유", "description": "밝은 에너지가 넘치는 곡"},
        {"title": "Walking On Sunshine", "artist": "Katrina and the Waves", "description": "신나는 80년대 팝"},
    ],
    "슬픔": [
        {"title": "Someone Like You", "artist": "Adele", "description": "이별의 아픔을 담은 발라드"},
        {"title": "서른 즈음에", "artist": "김광석", "description": "삶의 무게를 담은 명곡"},
        {"title": "Hurt", "artist": "Johnny Cash", "description": "깊은 슬픔과 후회의 노래"},
        {"title": "Spring Day", "artist": "BTS", "description": "그리움을 담은 감성적인 곡"},
    ],
    "분노": [
        {"title": "Killing In The Name", "artist": "Rage Against The Machine", "description": "사회적 분노를 표현한 록"},
        {"title": "Born Hater", "artist": "에픽하이", "description": "직설적이고 강렬한 랩"},
        {"title": "Break Stuff", "artist": "Limp Bizkit", "description": "분노를 폭발시키는 록"},
        {"title": "We Will Rock You", "artist": "Queen", "description": "강렬한 에너지의 록"},
    ],
    "평온": [
        {"title": "Clair de Lune", "artist": "Claude Debussy", "description": "달빛처럼 고요한 클래식"},
        {"title": "Don't Know Why", "artist": "Norah Jones", "description": "부드럽고 편안한 재즈"},
        {"title": "River Flows In You", "artist": "이루마", "description": "아름다운 피아노 선율"},
        {"title": "Weightless", "artist": "Marconi Union", "description": "과학적으로 이완 효과가 입증된 곡"},
    ],
    "불안": [
        {"title": "Creep", "artist": "Radiohead", "description": "불안과 소외감을 담은 명곡"},
        {"title": "Antifreeze", "artist": "검정치마", "description": "몽환적이고 불안한 감성"},
        {"title": "Mad World", "artist": "Gary Jules", "description": "우울하고 불안한 세상을 노래"},
        {"title": "Everybody Hurts", "artist": "R.E.M.", "description": "힘든 마음을 위로하는 곡"},
    ]
}

FOOD_DATA = {
    "행복": [
        {"name": "치킨", "description": "바삭하고 맛있는 치킨으로 기분 전환", "category": "야식"},
        {"name": "초밥", "description": "신선한 재료의 행복한 한 입", "category": "일식"},
        {"name": "피자", "description": "치즈가 가득한 행복", "category": "양식"},
        {"name": "케이크", "description": "달콤한 디저트로 기분 UP", "category": "디저트"},
    ],
    "슬픔": [
        {"name": "따뜻한 국밥", "description": "든든하고 위로가 되는 한 끼", "category": "한식"},
        {"name": "라면", "description": "뜨끈한 국물이 마음을 녹여줘요", "category": "간편식"},
        {"name": "죽", "description": "부드럽고 따뜻한 위로", "category": "한식"},
        {"name": "핫초코", "description": "달콤하고 따뜻한 음료", "category": "음료"},
    ],
    "분노": [
        {"name": "매운 떡볶이", "description": "매운맛으로 스트레스 해소", "category": "한식"},
        {"name": "불닭볶음면", "description": "화끈한 매운맛", "category": "간편식"},
        {"name": "마라탕", "description": "얼얼한 매운맛으로 기분 전환", "category": "중식"},
        {"name": "양꼬치", "artist": "향신료 가득한 중식", "category": "중식"},
    ],
    "평온": [
        {"name": "샐러드", "description": "신선하고 건강한 한 끼", "category": "건강식"},
        {"name": "연어 요리", "description": "담백하고 고급스러운 맛", "category": "양식"},
        {"name": "녹차", "description": "마음을 정화하는 차", "category": "음료"},
        {"name": "과일", "description": "자연의 달콤함", "category": "간식"},
    ],
    "불안": [
        {"name": "따뜻한 수프", "description": "마음을 진정시키는 따뜻한 한 그릇", "category": "양식"},
        {"name": "바나나", "description": "세로토닌을 높여주는 과일", "category": "과일"},
        {"name": "허브티", "description": "긴장을 완화시켜주는 차", "category": "음료"},
        {"name": "아보카도 토스트", "description": "건강하고 포만감 있는 식사", "category": "브런치"},
    ]
}

def get_recommendation_data(emotion: str, category: str):
    """감정과 카테고리에 따라 추천 데이터를 반환합니다."""
    data_map = {
        "도서": BOOK_DATA,
        "음악": MUSIC_DATA,
        "식사": FOOD_DATA
    }

    data_source = data_map.get(category, {})
    return data_source.get(emotion, [])
