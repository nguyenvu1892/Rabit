from brain.session_detector import get_session


def extract_context(df):

    session = get_session()

    return {
        "session": session
    }
