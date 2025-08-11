"""
Enterprise Customer Satisfaction (CSAT) Tracker for Chatbot Framework

This module provides comprehensive customer satisfaction tracking with:
- Real-time satisfaction scoring
- Multi-dimensional feedback analysis
- Trend monitoring and alerting
- Automated satisfaction surveys
"""

from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import asyncio
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

from .base_core import ICSATTracker, SentimentType, secure_logger


class SatisfactionLevel(Enum):
    """Customer satisfaction levels."""
    VERY_DISSATISFIED = 1
    DISSATISFIED = 2
    NEUTRAL = 3
    SATISFIED = 4
    VERY_SATISFIED = 5


class FeedbackType(Enum):
    """Types of customer feedback."""
    EXPLICIT = "explicit"  # Direct rating/feedback
    IMPLICIT = "implicit"  # Inferred from behavior
    SENTIMENT = "sentiment"  # From sentiment analysis
    BEHAVIORAL = "behavioral"  # From interaction patterns


class SurveyTrigger(Enum):
    """Triggers for satisfaction surveys."""
    SESSION_END = "session_end"
    ISSUE_RESOLVED = "issue_resolved"
    ESCALATION = "escalation"
    NEGATIVE_SENTIMENT = "negative_sentiment"
    PERIODIC = "periodic"
    RANDOM_SAMPLE = "random_sample"


@dataclass
class SatisfactionScore:
    """Individual satisfaction score record."""
    id: str
    session_id: str
    user_id: str
    score: float  # 1.0 to 5.0
    satisfaction_level: SatisfactionLevel
    feedback_type: FeedbackType
    confidence: float
    feedback_text: Optional[str]
    dimensions: Dict[str, float]  # helpfulness, accuracy, speed, etc.
    context: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'user_id': self.user_id,
            'score': self.score,
            'satisfaction_level': self.satisfaction_level.value,
            'feedback_type': self.feedback_type.value,
            'confidence': self.confidence,
            'feedback_text': self.feedback_text,
            'dimensions': self.dimensions,
            'context': self.context,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class SatisfactionTrend:
    """Satisfaction trend analysis."""
    period: str
    average_score: float
    score_distribution: Dict[SatisfactionLevel, int]
    trend_direction: str  # improving, declining, stable
    confidence_interval: Tuple[float, float]
    sample_size: int
    key_insights: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'period': self.period,
            'average_score': self.average_score,
            'score_distribution': {k.name: v for k, v in self.score_distribution.items()},
            'trend_direction': self.trend_direction,
            'confidence_interval': self.confidence_interval,
            'sample_size': self.sample_size,
            'key_insights': self.key_insights
        }


class SatisfactionCalculator:
    """Calculates satisfaction scores from various inputs."""
    
    def __init__(self):
        # Weights for different satisfaction dimensions
        self.dimension_weights = {
            'helpfulness': 0.3,
            'accuracy': 0.25,
            'speed': 0.2,
            'clarity': 0.15,
            'completeness': 0.1
        }
        
        # Sentiment to satisfaction mapping
        self.sentiment_mapping = {
            SentimentType.POSITIVE: 4.2,
            SentimentType.NEUTRAL: 3.0,
            SentimentType.NEGATIVE: 1.8
        }
    
    def calculate_from_sentiment(self, sentiment: SentimentType, 
                               confidence: float) -> Tuple[float, float]:
        """Calculate satisfaction score from sentiment analysis."""
        base_score = self.sentiment_mapping[sentiment]
        
        # Adjust score based on confidence
        if confidence > 0.8:
            # High confidence - use score as is
            final_score = base_score
            calc_confidence = 0.7
        elif confidence > 0.6:
            # Medium confidence - slight adjustment toward neutral
            final_score = base_score * 0.9 + 3.0 * 0.1
            calc_confidence = 0.5
        else:
            # Low confidence - more neutral
            final_score = base_score * 0.7 + 3.0 * 0.3
            calc_confidence = 0.3
        
        return final_score, calc_confidence
    
    def calculate_from_behavior(self, interaction_data: Dict[str, Any]) -> Tuple[float, float]:
        """Calculate satisfaction from behavioral indicators."""
        score_factors = []
        
        # Session duration (not too short, not too long)
        duration_minutes = interaction_data.get('duration_minutes', 0)
        if 2 <= duration_minutes <= 15:
            score_factors.append(4.0)  # Good duration
        elif duration_minutes < 1:
            score_factors.append(2.0)  # Too short - might be frustrated
        else:
            score_factors.append(3.0)  # Neutral
        
        # Number of clarification requests
        clarifications = interaction_data.get('clarification_count', 0)
        if clarifications == 0:
            score_factors.append(4.5)
        elif clarifications <= 2:
            score_factors.append(3.5)
        else:
            score_factors.append(2.0)
        
        # Task completion
        task_completed = interaction_data.get('task_completed', False)
        if task_completed:
            score_factors.append(4.5)
        else:
            score_factors.append(2.5)
        
        # Escalation to human
        escalated = interaction_data.get('escalated', False)
        if escalated:
            score_factors.append(2.0)
        else:
            score_factors.append(4.0)
        
        # Average the factors
        if score_factors:
            final_score = statistics.mean(score_factors)
            confidence = 0.6  # Behavioral inference has medium confidence
        else:
            final_score = 3.0
            confidence = 0.3
        
        return final_score, confidence
    
    def calculate_composite_score(self, scores: List[Tuple[float, float, FeedbackType]]) -> Tuple[float, float]:
        """Calculate composite satisfaction score from multiple inputs."""
        if not scores:
            return 3.0, 0.0
        
        # Weight different feedback types
        type_weights = {
            FeedbackType.EXPLICIT: 1.0,
            FeedbackType.SENTIMENT: 0.7,
            FeedbackType.BEHAVIORAL: 0.5,
            FeedbackType.IMPLICIT: 0.3
        }
        
        weighted_scores = []
        total_weight = 0
        
        for score, confidence, feedback_type in scores:
            weight = type_weights.get(feedback_type, 0.5) * confidence
            weighted_scores.append(score * weight)
            total_weight += weight
        
        if total_weight > 0:
            final_score = sum(weighted_scores) / total_weight
            final_confidence = min(1.0, total_weight / len(scores))
        else:
            final_score = 3.0
            final_confidence = 0.1
        
        return final_score, final_confidence


class SurveyManager:
    """Manages satisfaction surveys and feedback collection."""
    
    def __init__(self):
        self.survey_templates = {
            SurveyTrigger.SESSION_END: {
                'question': "How satisfied were you with our service today?",
                'scale': "Please rate from 1 (Very Dissatisfied) to 5 (Very Satisfied)",
                'follow_up': "Any additional feedback you'd like to share?"
            },
            SurveyTrigger.ISSUE_RESOLVED: {
                'question': "Did we successfully resolve your issue?",
                'scale': "Please rate your satisfaction with the resolution",
                'follow_up': "How could we have done better?"
            },
            SurveyTrigger.ESCALATION: {
                'question': "We apologize for the need to escalate. How was your overall experience?",
                'scale': "Your feedback helps us improve",
                'follow_up': "What could we have done differently?"
            },
            SurveyTrigger.NEGATIVE_SENTIMENT: {
                'question': "We noticed you might be having difficulties. How can we improve?",
                'scale': "Please help us understand your experience",
                'follow_up': "What would make this better for you?"
            }
        }
    
    def should_trigger_survey(self, trigger: SurveyTrigger, context: Dict[str, Any]) -> bool:
        """Determine if a survey should be triggered."""
        if trigger == SurveyTrigger.SESSION_END:
            # Trigger for sessions longer than 2 minutes
            return context.get('duration_minutes', 0) > 2
        
        elif trigger == SurveyTrigger.ISSUE_RESOLVED:
            # Trigger when a task is marked as completed
            return context.get('task_completed', False)
        
        elif trigger == SurveyTrigger.ESCALATION:
            # Always trigger after escalation
            return context.get('escalated', False)
        
        elif trigger == SurveyTrigger.NEGATIVE_SENTIMENT:
            # Trigger on strong negative sentiment
            sentiment_score = context.get('sentiment_score', 0)
            return sentiment_score < 2.0
        
        elif trigger == SurveyTrigger.RANDOM_SAMPLE:
            # Random sampling (10% of sessions)
            import random
            return random.random() < 0.1
        
        return False
    
    def generate_survey(self, trigger: SurveyTrigger, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a survey for the given trigger."""
        template = self.survey_templates.get(trigger, self.survey_templates[SurveyTrigger.SESSION_END])
        
        return {
            'trigger': trigger.value,
            'question': template['question'],
            'scale_description': template['scale'],
            'follow_up_question': template['follow_up'],
            'survey_id': f"survey_{trigger.value}_{datetime.utcnow().timestamp()}",
            'context': context
        }


class CSATTracker(ICSATTracker):
    """Enterprise CSAT tracker with comprehensive satisfaction monitoring."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the CSAT tracker."""
        self.config = config or {}
        self.calculator = SatisfactionCalculator()
        self.survey_manager = SurveyManager()
        
        # Configuration
        self.enable_surveys = self.config.get('enable_surveys', True)
        self.alert_threshold = self.config.get('alert_threshold', 2.5)
        self.trend_window_days = self.config.get('trend_window_days', 7)
        self.min_scores_for_trend = self.config.get('min_scores_for_trend', 10)
        
        # Data storage
        self.satisfaction_scores: List[SatisfactionScore] = []
        self.max_stored_scores = self.config.get('max_stored_scores', 1000)
        
        # Alert tracking
        self.alerts_sent: List[Dict[str, Any]] = []
        self.last_alert_time: Optional[datetime] = None
        self.alert_cooldown_minutes = self.config.get('alert_cooldown_minutes', 30)
        
        # Performance metrics
        self.metrics = {
            'total_scores_recorded': 0,
            'explicit_feedback_count': 0,
            'surveys_sent': 0,
            'surveys_completed': 0,
            'alerts_triggered': 0,
            'average_score_7d': 0.0,
            'satisfaction_distribution': {level.name: 0 for level in SatisfactionLevel}
        }
        
        secure_logger.info("CSATTracker initialized", extra={
            'surveys_enabled': self.enable_surveys,
            'alert_threshold': self.alert_threshold,
            'trend_window': self.trend_window_days
        })
    
    async def record_satisfaction(self, session_id: str, user_id: str, 
                                score: Optional[float] = None,
                                feedback_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Record customer satisfaction score.
        
        Args:
            session_id: Session identifier
            user_id: User identifier  
            score: Explicit satisfaction score (1-5)
            feedback_data: Additional feedback context
            
        Returns:
            Satisfaction record ID
        """
        try:
            feedback_data = feedback_data or {}
            
            # Determine feedback type and calculate score
            if score is not None:
                # Explicit feedback
                final_score = max(1.0, min(5.0, score))
                feedback_type = FeedbackType.EXPLICIT
                confidence = 0.9
            else:
                # Calculate from available data
                final_score, confidence = self._calculate_implicit_score(feedback_data)
                feedback_type = self._determine_feedback_type(feedback_data)
            
            # Determine satisfaction level
            satisfaction_level = self._score_to_level(final_score)
            
            # Extract dimensions
            dimensions = self._extract_dimensions(feedback_data)
            
            # Create satisfaction record
            satisfaction_record = SatisfactionScore(
                id=f"csat_{session_id}_{user_id}_{datetime.utcnow().timestamp()}",
                session_id=session_id,
                user_id=user_id,
                score=final_score,
                satisfaction_level=satisfaction_level,
                feedback_type=feedback_type,
                confidence=confidence,
                feedback_text=feedback_data.get('feedback_text'),
                dimensions=dimensions,
                context=feedback_data.get('context', {}),
                timestamp=datetime.utcnow()
            )
            
            # Store the record
            self.satisfaction_scores.append(satisfaction_record)
            
            # Maintain storage limits
            if len(self.satisfaction_scores) > self.max_stored_scores:
                self.satisfaction_scores = self.satisfaction_scores[-self.max_stored_scores:]
            
            # Update metrics
            self._update_metrics(satisfaction_record)
            
            # Check for alerts
            await self._check_alerts(satisfaction_record)
            
            # Trigger survey if appropriate
            if self.enable_surveys:
                await self._maybe_trigger_survey(satisfaction_record, feedback_data)
            
            secure_logger.info("Satisfaction recorded", extra={
                'record_id': satisfaction_record.id,
                'session_id': session_id,
                'user_id': user_id,
                'score': final_score,
                'level': satisfaction_level.name,
                'feedback_type': feedback_type.value,
                'confidence': confidence
            })
            
            return satisfaction_record.id
            
        except Exception as e:
            secure_logger.error(f"Error recording satisfaction: {str(e)}", extra={
                'session_id': session_id,
                'user_id': user_id,
                'error_type': type(e).__name__
            })
            raise
    
    def _calculate_implicit_score(self, feedback_data: Dict[str, Any]) -> Tuple[float, float]:
        """Calculate satisfaction score from implicit feedback."""
        scores = []
        
        # From sentiment analysis
        if 'sentiment' in feedback_data:
            sentiment = SentimentType(feedback_data['sentiment'])
            sentiment_confidence = feedback_data.get('sentiment_confidence', 0.5)
            score, confidence = self.calculator.calculate_from_sentiment(sentiment, sentiment_confidence)
            scores.append((score, confidence, FeedbackType.SENTIMENT))
        
        # From behavioral data
        if 'interaction_data' in feedback_data:
            score, confidence = self.calculator.calculate_from_behavior(feedback_data['interaction_data'])
            scores.append((score, confidence, FeedbackType.BEHAVIORAL))
        
        # Calculate composite score
        if scores:
            return self.calculator.calculate_composite_score(scores)
        else:
            return 3.0, 0.3  # Default neutral score
    
    def _determine_feedback_type(self, feedback_data: Dict[str, Any]) -> FeedbackType:
        """Determine the primary feedback type."""
        if 'explicit_rating' in feedback_data:
            return FeedbackType.EXPLICIT
        elif 'sentiment' in feedback_data:
            return FeedbackType.SENTIMENT
        elif 'interaction_data' in feedback_data:
            return FeedbackType.BEHAVIORAL
        else:
            return FeedbackType.IMPLICIT
    
    def _score_to_level(self, score: float) -> SatisfactionLevel:
        """Convert numeric score to satisfaction level."""
        if score >= 4.5:
            return SatisfactionLevel.VERY_SATISFIED
        elif score >= 3.5:
            return SatisfactionLevel.SATISFIED
        elif score >= 2.5:
            return SatisfactionLevel.NEUTRAL
        elif score >= 1.5:
            return SatisfactionLevel.DISSATISFIED
        else:
            return SatisfactionLevel.VERY_DISSATISFIED
    
    def _extract_dimensions(self, feedback_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract satisfaction dimensions from feedback data."""
        dimensions = {}
        
        # Default dimensions with neutral scores
        for dimension in self.calculator.dimension_weights.keys():
            dimensions[dimension] = 3.0
        
        # Override with explicit dimension scores if provided
        if 'dimensions' in feedback_data:
            dimensions.update(feedback_data['dimensions'])
        
        # Infer from interaction data
        interaction_data = feedback_data.get('interaction_data', {})
        
        # Speed dimension
        response_time = interaction_data.get('average_response_time', 2.0)
        if response_time < 1.0:
            dimensions['speed'] = 5.0
        elif response_time < 3.0:
            dimensions['speed'] = 4.0
        elif response_time > 10.0:
            dimensions['speed'] = 2.0
        
        # Completeness dimension
        if interaction_data.get('task_completed', False):
            dimensions['completeness'] = 4.5
        elif interaction_data.get('partial_completion', False):
            dimensions['completeness'] = 3.0
        else:
            dimensions['completeness'] = 2.0
        
        return dimensions
    
    async def _check_alerts(self, record: SatisfactionScore) -> None:
        """Check if satisfaction alerts should be triggered."""
        try:
            # Don't alert if we're in cooldown period
            if (self.last_alert_time and 
                datetime.utcnow() - self.last_alert_time < timedelta(minutes=self.alert_cooldown_minutes)):
                return
            
            # Alert on very low individual scores
            if record.score < self.alert_threshold and record.confidence > 0.6:
                await self._send_alert({
                    'type': 'low_individual_score',
                    'score': record.score,
                    'session_id': record.session_id,
                    'user_id': record.user_id,
                    'timestamp': record.timestamp.isoformat()
                })
            
            # Alert on trend deterioration
            recent_trend = await self.get_satisfaction_trend(hours=24)
            if (recent_trend['sample_size'] >= 5 and 
                recent_trend['average_score'] < self.alert_threshold):
                await self._send_alert({
                    'type': 'declining_trend',
                    'average_score': recent_trend['average_score'],
                    'sample_size': recent_trend['sample_size'],
                    'trend_direction': recent_trend.get('trend_direction', 'unknown')
                })
                
        except Exception as e:
            secure_logger.error(f"Error checking satisfaction alerts: {str(e)}")
    
    async def _send_alert(self, alert_data: Dict[str, Any]) -> None:
        """Send satisfaction alert."""
        alert_record = {
            'alert_id': f"alert_{datetime.utcnow().timestamp()}",
            'timestamp': datetime.utcnow().isoformat(),
            **alert_data
        }
        
        self.alerts_sent.append(alert_record)
        self.last_alert_time = datetime.utcnow()
        self.metrics['alerts_triggered'] += 1
        
        secure_logger.warning("CSAT Alert triggered", extra=alert_record)
    
    async def _maybe_trigger_survey(self, record: SatisfactionScore, 
                                  feedback_data: Dict[str, Any]) -> None:
        """Check if a survey should be triggered."""
        try:
            context = {
                'satisfaction_score': record.score,
                'session_id': record.session_id,
                **feedback_data.get('context', {}),
                **feedback_data.get('interaction_data', {})
            }
            
            # Check different survey triggers
            triggers_to_check = [
                SurveyTrigger.SESSION_END,
                SurveyTrigger.ISSUE_RESOLVED,
                SurveyTrigger.ESCALATION,
                SurveyTrigger.NEGATIVE_SENTIMENT,
                SurveyTrigger.RANDOM_SAMPLE
            ]
            
            for trigger in triggers_to_check:
                if self.survey_manager.should_trigger_survey(trigger, context):
                    survey = self.survey_manager.generate_survey(trigger, context)
                    
                    # In a real implementation, this would send the survey
                    # For now, we'll just log it
                    secure_logger.info("Survey triggered", extra={
                        'survey_id': survey['survey_id'],
                        'trigger': trigger.value,
                        'session_id': record.session_id,
                        'user_id': record.user_id
                    })
                    
                    self.metrics['surveys_sent'] += 1
                    break  # Only trigger one survey per session
                    
        except Exception as e:
            secure_logger.error(f"Error triggering survey: {str(e)}")
    
    def _update_metrics(self, record: SatisfactionScore) -> None:
        """Update performance metrics."""
        self.metrics['total_scores_recorded'] += 1
        self.metrics['satisfaction_distribution'][record.satisfaction_level.name] += 1
        
        if record.feedback_type == FeedbackType.EXPLICIT:
            self.metrics['explicit_feedback_count'] += 1
        
        # Update 7-day average
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        recent_scores = [
            s.score for s in self.satisfaction_scores 
            if s.timestamp >= seven_days_ago
        ]
        
        if recent_scores:
            self.metrics['average_score_7d'] = statistics.mean(recent_scores)
    
    async def get_satisfaction_trend(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get satisfaction trend analysis for the specified time period.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Trend analysis results
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Filter scores within time window
            recent_scores = [
                score for score in self.satisfaction_scores
                if score.timestamp >= cutoff_time
            ]
            
            if len(recent_scores) < self.min_scores_for_trend:
                return {
                    'period': f'{hours} hours',
                    'average_score': 0.0,
                    'sample_size': len(recent_scores),
                    'trend_direction': 'insufficient_data',
                    'confidence_interval': (0.0, 0.0),
                    'score_distribution': {},
                    'key_insights': ['Insufficient data for trend analysis']
                }
            
            # Calculate statistics
            scores = [s.score for s in recent_scores]
            average_score = statistics.mean(scores)
            
            # Score distribution
            distribution = {}
            for level in SatisfactionLevel:
                distribution[level] = sum(1 for s in recent_scores if s.satisfaction_level == level)
            
            # Trend direction (compare first half vs second half)
            mid_point = len(scores) // 2
            if mid_point > 0:
                first_half_avg = statistics.mean(scores[:mid_point])
                second_half_avg = statistics.mean(scores[mid_point:])
                
                if second_half_avg > first_half_avg + 0.2:
                    trend_direction = 'improving'
                elif second_half_avg < first_half_avg - 0.2:
                    trend_direction = 'declining'
                else:
                    trend_direction = 'stable'
            else:
                trend_direction = 'stable'
            
            # Confidence interval (simple approach)
            std_dev = statistics.stdev(scores) if len(scores) > 1 else 0
            margin = 1.96 * (std_dev / (len(scores) ** 0.5))  # 95% confidence
            confidence_interval = (average_score - margin, average_score + margin)
            
            # Generate insights
            insights = []
            if average_score < 2.5:
                insights.append("Below average satisfaction - immediate attention needed")
            elif average_score > 4.0:
                insights.append("High satisfaction levels - maintain current performance")
            
            if distribution.get(SatisfactionLevel.VERY_DISSATISFIED, 0) > len(recent_scores) * 0.2:
                insights.append("High proportion of very dissatisfied customers")
            
            return {
                'period': f'{hours} hours',
                'average_score': average_score,
                'sample_size': len(recent_scores),
                'trend_direction': trend_direction,
                'confidence_interval': confidence_interval,
                'score_distribution': distribution,
                'key_insights': insights
            }
            
        except Exception as e:
            secure_logger.error(f"Error calculating satisfaction trend: {str(e)}")
            return {
                'period': f'{hours} hours',
                'error': str(e),
                'sample_size': 0
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get CSAT tracker performance metrics."""
        completion_rate = (
            self.metrics['surveys_completed'] / self.metrics['surveys_sent']
            if self.metrics['surveys_sent'] > 0 else 0
        )
        
        return {
            **self.metrics,
            'survey_completion_rate': completion_rate,
            'total_stored_scores': len(self.satisfaction_scores),
            'alert_frequency': len(self.alerts_sent),
            'recent_alerts': self.alerts_sent[-5:] if self.alerts_sent else []
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the CSAT tracker."""
        try:
            # Test satisfaction recording
            test_session_id = "test_session"
            test_user_id = "test_user"
            
            record_id = await self.record_satisfaction(
                session_id=test_session_id,
                user_id=test_user_id,
                score=4.0,
                feedback_data={'feedback_text': 'Test feedback'}
            )
            
            # Remove test record
            self.satisfaction_scores = [
                s for s in self.satisfaction_scores 
                if s.id != record_id
            ]
            
            return {
                'status': 'healthy',
                'test_successful': record_id is not None,
                'metrics': self.get_metrics()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'metrics': self.get_metrics()
            }


# Export the main classes
__all__ = ['CSATTracker', 'SatisfactionScore', 'SatisfactionLevel', 'SatisfactionTrend']
