"""
Cold start recommendation module for handling new users with little history.
"""
import logging
from typing import Dict, List, Any, Optional
from . import data_fetcher

logger = logging.getLogger(__name__)

async def generate_cold_start_recommendations(user_profile: Dict[str, Any], limit: int = 5) -> Dict[str, Any]:
    """
    Generate recommendations for new users with little history (cold start)
    
    Args:
        user_profile: The user profile data
        limit: The maximum number of recommendations to generate
        
    Returns:
        A dictionary containing the recommended tests and lectures
    """
    logger.info(f"Generating cold start recommendations for user ID: {user_profile['userId']}")
    
    try:
        # Get test and lecture candidates
        test_candidates = await data_fetcher.get_test_candidates()
        lecture_candidates = await data_fetcher.get_lecture_candidates()
        
        # For cold start, recommend popular items (high number of user attempts)
        # and beginner level content (lower difficulty)
        recommended_tests = await recommend_cold_start_tests(user_profile, test_candidates, limit)
        recommended_lectures = await recommend_cold_start_lectures(user_profile, lecture_candidates, limit)
        # Format output with detailed explanation
        test_recommendations = []
        for rec in recommended_tests:
            # Create explanation for why this test was recommended
            explanation_parts = []
            # topic_list = ", ".join(rec.get("topics", []))
            # explanation_parts.append(f"Liên quan đến các chủ đề: {topic_list}")
            
            diff_gap = abs(rec.get("difficulty", 0) - user_profile.get("target", 0))
            explanation_parts.append(f"Độ khó {rec.get('difficulty', 0)}, cách mục tiêu {diff_gap} điểm")
            
            explanation = " | ".join(explanation_parts)

            test_recommendations.append({
                "id": rec.get("id", ""),
                "name": rec.get("name", ""),
                "score": rec.get("score", 0),
                "explanation": explanation,
                "testId": rec.get("testId", "")
            })

        lecture_recommendations = []
        for rec in recommended_lectures:
            # Create explanation for why this lesson was recommended
            explanation_parts = []
            topic_list = ", ".join(rec.get("topics", []))
            explanation_parts.append(f"Giúp cải thiện các chủ đề: {topic_list}")
            
            explanation = " | ".join(explanation_parts)

            lecture_recommendations.append({
                "id": rec.get("id", ""),
                "name": rec.get("name", ""),
                "score": rec.get("score", 0),
                "explanation": explanation,
                "lectureId": rec.get("lectureId", "")
            })

        return {
            "userId": user_profile["userId"],
            "recommendedTests": test_recommendations,
            "recommendedLectures": lecture_recommendations
        }
    
    except Exception as e:
        logger.error(f"Error generating cold start recommendations: {str(e)}")
        raise

async def recommend_cold_start_tests(user_profile: Dict[str, Any], 
                              test_candidates: List[Dict[str, Any]], 
                              limit: int) -> List[Dict[str, Any]]:
    """
    Recommend tests for a new user
    
    Args:
        user_profile: The user profile data
        test_candidates: The list of test candidates
        limit: The maximum number of recommendations to generate
        
    Returns:
        A list of recommended tests
    """
    # Filter out any tests the user has already taken
    completed_test_ids = [item["testId"] for item in user_profile.get("testHistory", [])]
    available_tests = [test for test in test_candidates if test["testId"] not in completed_test_ids]
    
    # For new users, recommend:
    # 1. Tests with lower difficulty
    # 2. Popular tests (high number of user attempts)
    
    # Copy and sort by a combined score
    target_difficulty = user_profile.get("target", 500) // 100  # Normalize target to difficulty scale
    
    scored_tests = []
    for test in available_tests:
        # Calculate popularity score (normalized by max attempts)
        max_attempts = max([t["totalUserAttempt"] for t in test_candidates]) if test_candidates else 1
        popularity_score = test["totalUserAttempt"] / max_attempts if max_attempts > 0 else 0
        
        # Calculate difficulty match score (closer to target is better)
        # Normalize to a 0-1 scale where 1 is perfect match
        difficulty_diff = abs(test["difficulty"] - target_difficulty) / 10  # Assume max difference is 10
        difficulty_score = 1 - min(1.0, difficulty_diff)  # Invert so higher is better
        
        # Combined score (weight popularity more for cold start)
        combined_score = 0.7 * popularity_score + 0.3 * difficulty_score
        
        scored_tests.append({
            "id": test["testId"],
            "name": test["name"],
            "score": combined_score,
            "already_taken": False,
            "attempts": 0,
            "avg_score": 0,
            "difficulty": test["difficulty"],
            "topics": test["topics"],
            "testId": test["testId"]
        })
    
    # Sort by combined score (descending)
    scored_tests.sort(key=lambda x: x["score"], reverse=True)
    
    # Return top N recommendations
    return scored_tests[:limit]

async def recommend_cold_start_lectures(user_profile: Dict[str, Any], 
                                 lecture_candidates: List[Dict[str, Any]], 
                                 limit: int) -> List[Dict[str, Any]]:
    """
    Recommend lectures for a new user
    
    Args:
        user_profile: The user profile data
        lecture_candidates: The list of lecture candidates
        limit: The maximum number of recommendations to generate
        
    Returns:
        A list of recommended lectures
    """
    # Filter out lectures the user has already started
    started_lecture_ids = list(user_profile.get("learningProgress", {}).keys())
    available_lectures = [lecture for lecture in lecture_candidates 
                         if lecture["lectureId"] not in started_lecture_ids]
    
    # For new users, recommend:
    # 1. Basic/introductory lectures
    # 2. Newer lectures
    
    # Calculate a score for each lecture based on recency
    # Assume newer lectures are better for beginners
    scored_lectures = []
    for lecture in available_lectures:
        # Add a basic score (could be enhanced with more sophisticated logic)
        scored_lectures.append({
            "id": lecture["lectureId"],
            "name": lecture["name"],
            "score": 1.0,
            "already_learned": False,
            "completion": 0,
            "topics": lecture["topics"],
            "lectureId": lecture["lectureId"]
        })
    
    # Return top N recommendations
    return scored_lectures[:limit]