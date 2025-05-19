"""
FastAPI Recommender Service Entry Point
"""
from fastapi import FastAPI, HTTPException, Depends, Header, Query
from typing import List, Optional, Dict, Any
import logging
from pydantic import BaseModel

from app.logic import data_fetcher
from app.logic.core_recommend import recommend_hybrid, is_cold_start_user
from app.logic.cold_start import generate_cold_start_recommendations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app instance
app = FastAPI(
    title="TOEIC Practice Recommender API",
    description="API for generating personalized TOEIC test and lecture recommendations",
    version="1.0.0"
)

# Pydantic models for request and response
class RecommendRequest(BaseModel):
    userId: str

class RecommendResponse(BaseModel):
    userId: str
    recommendedTests: List[Dict[str, Any]]
    recommendedLectures: List[Dict[str, Any]]

@app.get("/")
async def root():
    """Root endpoint that returns API information"""
    return {
        "name": "TOEIC Practice Recommender API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/recommendations/{user_id}")
async def get_recommendations(
    user_id: str,
    limit: int = Query(5, description="Maximum number of recommendations to return")
):
    """
    Generate personalized recommendations for a user
    
    Args:
        user_id: The ID of the user to generate recommendations for
        limit: The maximum number of recommendations to return
        
    Returns:
        A dictionary containing the recommended tests and lectures
    """
    logger.info(f"Processing recommendation request for user: {user_id}")
    try:
        # Fetch the user profile
        user_profile_api_response = await data_fetcher.get_user_profile(user_id)
        user_profile = user_profile_api_response.get('data', {})
        logger.info(f"Success get user profile id {user_id}")
        # Check if this is a cold start user (no history)
        is_cold_start = is_cold_start_user(user_profile)
        
        if is_cold_start:
            logger.info(f"Using cold start recommendations for new user: {user_id}")
            result = await generate_cold_start_recommendations(user_profile, limit)
            return result
        else:
            logger.info(f"Generating personalized recommendations for user: {user_id}")
            hybrid_result = await recommend_hybrid(user_profile, limit)
            return hybrid_result
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 