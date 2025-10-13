import pickle

# 감정 목록
EMOTIONS = ["joy", "sadness", "anger", "fear", "calm"]

# 각 감정에 대응하는 더미 콘텐츠 데이터
EMOTION_DATA = {
    "joy": {
        "music": [
            "Happy - Pharrell Williams",
            "Walking on Sunshine - Katrina & The Waves",
            "Good as Hell - Lizzo"
        ],
        "book": [
            "The Happiness Project - Gretchen Rubin",
            "Joyful - Ingrid Fetell Lee"
        ],
        "movie": [
            "Inside Out",
            "Amélie"
        ]
    },
    "sadness": {
        "music": [
            "Someone Like You - Adele",
            "Fix You - Coldplay",
            "Let Her Go - Passenger"
        ],
        "book": [
            "The Fault in Our Stars - John Green",
            "Norwegian Wood - Haruki Murakami"
        ],
        "movie": [
            "Blue Valentine",
            "Manchester by the Sea"
        ]
    },
    "anger": {
        "music": [
            "Break Stuff - Limp Bizkit",
            "Killing in the Name - Rage Against the Machine",
            "Smells Like Teen Spirit - Nirvana"
        ],
        "book": [
            "Anger - Thich Nhat Hanh",
            "Rage - Bob Woodward"
        ],
        "movie": [
            "Joker",
            "Falling Down"
        ]
    },
    "fear": {
        "music": [
            "Disturbia - Rihanna",
            "Bury a Friend - Billie Eilish",
            "Haunted - Beyoncé"
        ],
        "book": [
            "It - Stephen King",
            "The Shining - Stephen King"
        ],
        "movie": [
            "Get Out",
            "A Quiet Place"
        ]
    },
    "calm": {
        "music": [
            "Weightless - Marconi Union",
            "Bloom - The Paper Kites",
            "River Flows in You - Yiruma"
        ],
        "book": [
            "The Things You Can See Only When You Slow Down - Haemin Sunim",
            "The Book of Joy - Dalai Lama"
        ],
        "movie": [
            "Call Me by Your Name",
            "The Secret Life of Walter Mitty"
        ]
    }
}

# pkl 파일로 저장
with open("emotion_data.pkl", "wb") as f:
    pickle.dump({"emotions": EMOTIONS, "data": EMOTION_DATA}, f)

print("✅ emotion_data.pkl 파일이 성공적으로 생성되었습니다.")
