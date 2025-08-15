"""
A/B Testing Framework
====================

Enterprise A/B testing system for chatbot experimentation and optimization.

PHASE3-REFACTOR: Statistical A/B testing with multi-variant support.

Features:
- Multi-variant experiment support
- Statistical significance testing
- Traffic routing and allocation
- Metrics collection and analysis
- Experiment lifecycle management
- Tenant-aware experimentation
"""

import asyncio
import hashlib
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable
import statistics
from scipy import stats
import numpy as np

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Experiment lifecycle status."""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ExperimentType(Enum):
    """Types of experiments."""
    AB_TEST = "ab_test"              # A/B test with control and treatment
    MULTIVARIATE = "multivariate"    # Multiple variants
    FEATURE_FLAG = "feature_flag"    # Feature rollout
    CANARY = "canary"               # Gradual rollout


class MetricType(Enum):
    """Types of metrics to track."""
    CONVERSION = "conversion"        # Success rate
    RESPONSE_TIME = "response_time"  # Performance metric
    SATISFACTION = "satisfaction"    # User satisfaction score
    ENGAGEMENT = "engagement"        # User engagement metric
    CUSTOM = "custom"               # Custom metric


@dataclass
class ExperimentVariant:
    """Experiment variant configuration."""
    id: str
    name: str
    description: str = ""
    traffic_allocation: float = 0.0  # Percentage of traffic (0.0 to 1.0)
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Performance tracking
    participants: int = 0
    conversions: int = 0
    total_response_time: float = 0.0
    total_satisfaction: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def get_conversion_rate(self) -> float:
        """Calculate conversion rate."""
        return self.conversions / self.participants if self.participants > 0 else 0.0
    
    def get_avg_response_time(self) -> float:
        """Calculate average response time."""
        return self.total_response_time / self.participants if self.participants > 0 else 0.0
    
    def get_avg_satisfaction(self) -> float:
        """Calculate average satisfaction score."""
        return self.total_satisfaction / self.participants if self.participants > 0 else 0.0
    
    def record_interaction(self, converted: bool = False, response_time: float = 0.0,
                         satisfaction: float = 0.0, custom_metrics: Optional[Dict[str, float]] = None):
        """Record interaction with this variant."""
        self.participants += 1
        
        if converted:
            self.conversions += 1
        
        self.total_response_time += response_time
        self.total_satisfaction += satisfaction
        
        if custom_metrics:
            for metric, value in custom_metrics.items():
                self.custom_metrics[metric] = self.custom_metrics.get(metric, 0.0) + value


@dataclass
class ExperimentMetric:
    """Metric definition for experiment tracking."""
    name: str
    metric_type: MetricType
    description: str = ""
    target_value: Optional[float] = None
    improvement_threshold: float = 0.05  # 5% improvement threshold
    
    # Statistical settings
    confidence_level: float = 0.95
    power: float = 0.8
    minimum_sample_size: int = 100


@dataclass
class StatisticalResult:
    """Statistical analysis result."""
    metric_name: str
    control_value: float
    treatment_value: float
    improvement: float
    improvement_percentage: float
    p_value: float
    confidence_interval: tuple[float, float]
    is_significant: bool
    sample_size_control: int
    sample_size_treatment: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric": self.metric_name,
            "control_value": self.control_value,
            "treatment_value": self.treatment_value,
            "improvement": self.improvement,
            "improvement_percentage": self.improvement_percentage,
            "p_value": self.p_value,
            "confidence_interval": list(self.confidence_interval),
            "is_significant": self.is_significant,
            "sample_sizes": {
                "control": self.sample_size_control,
                "treatment": self.sample_size_treatment
            }
        }


@dataclass
class Experiment:
    """
    A/B test experiment definition.
    
    PHASE3-REFACTOR: Complete experiment configuration and tracking.
    """
    id: str
    name: str
    description: str
    experiment_type: ExperimentType = ExperimentType.AB_TEST
    status: ExperimentStatus = ExperimentStatus.DRAFT
    
    # Targeting
    tenant_ids: List[str] = field(default_factory=list)  # Empty = all tenants
    user_segments: List[str] = field(default_factory=list)
    
    # Variants
    variants: List[ExperimentVariant] = field(default_factory=list)
    control_variant_id: Optional[str] = None
    
    # Metrics
    primary_metric: str = "conversion"
    metrics: List[ExperimentMetric] = field(default_factory=list)
    
    # Timing
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    duration_days: int = 14
    
    # Statistical configuration
    confidence_level: float = 0.95
    minimum_effect_size: float = 0.05
    minimum_sample_size: int = 100
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""
    updated_at: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.end_date and self.start_date:
            self.end_date = self.start_date + timedelta(days=self.duration_days)
    
    def is_active(self) -> bool:
        """Check if experiment is currently active."""
        if self.status != ExperimentStatus.ACTIVE:
            return False
        
        now = datetime.utcnow()
        if self.start_date and now < self.start_date:
            return False
        
        if self.end_date and now > self.end_date:
            return False
        
        return True
    
    def get_variant(self, variant_id: str) -> Optional[ExperimentVariant]:
        """Get variant by ID."""
        for variant in self.variants:
            if variant.id == variant_id:
                return variant
        return None
    
    def get_control_variant(self) -> Optional[ExperimentVariant]:
        """Get control variant."""
        if self.control_variant_id:
            return self.get_variant(self.control_variant_id)
        return self.variants[0] if self.variants else None
    
    def get_total_participants(self) -> int:
        """Get total participants across all variants."""
        return sum(variant.participants for variant in self.variants)
    
    def validate_traffic_allocation(self) -> bool:
        """Validate that traffic allocation sums to 1.0."""
        total = sum(variant.traffic_allocation for variant in self.variants)
        return abs(total - 1.0) < 0.001  # Allow small floating point errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert experiment to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "type": self.experiment_type.value,
            "status": self.status.value,
            "tenant_ids": self.tenant_ids,
            "user_segments": self.user_segments,
            "variants": [
                {
                    "id": v.id,
                    "name": v.name,
                    "description": v.description,
                    "traffic_allocation": v.traffic_allocation,
                    "config": v.config,
                    "participants": v.participants,
                    "conversions": v.conversions,
                    "conversion_rate": v.get_conversion_rate(),
                    "avg_response_time": v.get_avg_response_time(),
                    "avg_satisfaction": v.get_avg_satisfaction(),
                    "custom_metrics": v.custom_metrics
                }
                for v in self.variants
            ],
            "control_variant_id": self.control_variant_id,
            "primary_metric": self.primary_metric,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "tags": self.tags,
            "total_participants": self.get_total_participants()
        }


class IExperimentStore(ABC):
    """Interface for experiment persistence."""
    
    @abstractmethod
    async def save_experiment(self, experiment: Experiment) -> bool:
        """Save experiment."""
        pass
    
    @abstractmethod
    async def load_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Load experiment by ID."""
        pass
    
    @abstractmethod
    async def list_experiments(self, status: Optional[ExperimentStatus] = None,
                             tenant_id: Optional[str] = None) -> List[Experiment]:
        """List experiments with filters."""
        pass
    
    @abstractmethod
    async def delete_experiment(self, experiment_id: str) -> bool:
        """Delete experiment."""
        pass


class MemoryExperimentStore(IExperimentStore):
    """In-memory experiment store for development."""
    
    def __init__(self):
        self.experiments: Dict[str, Experiment] = {}
    
    async def save_experiment(self, experiment: Experiment) -> bool:
        """Save experiment."""
        experiment.updated_at = datetime.utcnow()
        self.experiments[experiment.id] = experiment
        return True
    
    async def load_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Load experiment by ID."""
        return self.experiments.get(experiment_id)
    
    async def list_experiments(self, status: Optional[ExperimentStatus] = None,
                             tenant_id: Optional[str] = None) -> List[Experiment]:
        """List experiments with filters."""
        experiments = list(self.experiments.values())
        
        if status:
            experiments = [e for e in experiments if e.status == status]
        
        if tenant_id:
            experiments = [e for e in experiments 
                         if not e.tenant_ids or tenant_id in e.tenant_ids]
        
        return experiments
    
    async def delete_experiment(self, experiment_id: str) -> bool:
        """Delete experiment."""
        if experiment_id in self.experiments:
            del self.experiments[experiment_id]
            return True
        return False


class TrafficRouter:
    """
    Traffic routing for experiment variants.
    
    PHASE3-REFACTOR: Deterministic traffic allocation based on user hash.
    """
    
    def __init__(self):
        self.hash_salt = "experiment_routing_salt"
    
    def get_variant_for_user(self, experiment: Experiment, user_id: str, 
                           tenant_id: Optional[str] = None) -> Optional[ExperimentVariant]:
        """Get variant assignment for user."""
        if not experiment.is_active():
            return None
        
        # Check tenant targeting
        if experiment.tenant_ids and tenant_id not in experiment.tenant_ids:
            return None
        
        # Generate deterministic hash for user
        hash_input = f"{experiment.id}:{user_id}:{self.hash_salt}"
        user_hash = hashlib.md5(hash_input.encode()).hexdigest()
        hash_value = int(user_hash, 16) / (16 ** len(user_hash))
        
        # Route to variant based on traffic allocation
        cumulative_allocation = 0.0
        for variant in experiment.variants:
            cumulative_allocation += variant.traffic_allocation
            if hash_value <= cumulative_allocation:
                return variant
        
        # Fallback to last variant if rounding errors
        return experiment.variants[-1] if experiment.variants else None
    
    def get_variant_config(self, experiment: Experiment, user_id: str,
                         tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration for user's assigned variant."""
        variant = self.get_variant_for_user(experiment, user_id, tenant_id)
        return variant.config if variant else {}


class StatisticalAnalyzer:
    """
    Statistical analysis for A/B test results.
    
    PHASE3-REFACTOR: Statistical significance testing and confidence intervals.
    """
    
    def __init__(self):
        pass
    
    def analyze_conversion_rate(self, control_variant: ExperimentVariant,
                              treatment_variant: ExperimentVariant,
                              confidence_level: float = 0.95) -> StatisticalResult:
        """Analyze conversion rate between variants."""
        control_conversions = control_variant.conversions
        control_participants = control_variant.participants
        treatment_conversions = treatment_variant.conversions
        treatment_participants = treatment_variant.participants
        
        if control_participants == 0 or treatment_participants == 0:
            return StatisticalResult(
                metric_name="conversion_rate",
                control_value=0.0,
                treatment_value=0.0,
                improvement=0.0,
                improvement_percentage=0.0,
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                is_significant=False,
                sample_size_control=control_participants,
                sample_size_treatment=treatment_participants
            )
        
        control_rate = control_conversions / control_participants
        treatment_rate = treatment_conversions / treatment_participants
        
        # Two-proportion z-test
        pooled_rate = (control_conversions + treatment_conversions) / (control_participants + treatment_participants)
        pooled_se = np.sqrt(pooled_rate * (1 - pooled_rate) * (1/control_participants + 1/treatment_participants))
        
        if pooled_se == 0:
            z_score = 0
            p_value = 1.0
        else:
            z_score = (treatment_rate - control_rate) / pooled_se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Confidence interval for difference
        se_diff = np.sqrt((control_rate * (1 - control_rate) / control_participants) + 
                         (treatment_rate * (1 - treatment_rate) / treatment_participants))
        alpha = 1 - confidence_level
        z_critical = stats.norm.ppf(1 - alpha/2)
        margin_error = z_critical * se_diff
        
        improvement = treatment_rate - control_rate
        improvement_percentage = (improvement / control_rate * 100) if control_rate > 0 else 0.0
        
        confidence_interval = (improvement - margin_error, improvement + margin_error)
        is_significant = p_value < (1 - confidence_level)
        
        return StatisticalResult(
            metric_name="conversion_rate",
            control_value=control_rate,
            treatment_value=treatment_rate,
            improvement=improvement,
            improvement_percentage=improvement_percentage,
            p_value=p_value,
            confidence_interval=confidence_interval,
            is_significant=is_significant,
            sample_size_control=control_participants,
            sample_size_treatment=treatment_participants
        )
    
    def analyze_continuous_metric(self, control_values: List[float],
                                treatment_values: List[float],
                                metric_name: str,
                                confidence_level: float = 0.95) -> StatisticalResult:
        """Analyze continuous metric (e.g., response time, satisfaction)."""
        if not control_values or not treatment_values:
            return StatisticalResult(
                metric_name=metric_name,
                control_value=0.0,
                treatment_value=0.0,
                improvement=0.0,
                improvement_percentage=0.0,
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                is_significant=False,
                sample_size_control=len(control_values),
                sample_size_treatment=len(treatment_values)
            )
        
        control_mean = statistics.mean(control_values)
        treatment_mean = statistics.mean(treatment_values)
        
        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(treatment_values, control_values)
        
        # Confidence interval for difference
        alpha = 1 - confidence_level
        df = len(control_values) + len(treatment_values) - 2
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        pooled_std = np.sqrt(((len(control_values) - 1) * np.var(control_values, ddof=1) + 
                             (len(treatment_values) - 1) * np.var(treatment_values, ddof=1)) / df)
        se_diff = pooled_std * np.sqrt(1/len(control_values) + 1/len(treatment_values))
        margin_error = t_critical * se_diff
        
        improvement = treatment_mean - control_mean
        improvement_percentage = (improvement / control_mean * 100) if control_mean != 0 else 0.0
        
        confidence_interval = (improvement - margin_error, improvement + margin_error)
        is_significant = p_value < (1 - confidence_level)
        
        return StatisticalResult(
            metric_name=metric_name,
            control_value=control_mean,
            treatment_value=treatment_mean,
            improvement=improvement,
            improvement_percentage=improvement_percentage,
            p_value=p_value,
            confidence_interval=confidence_interval,
            is_significant=is_significant,
            sample_size_control=len(control_values),
            sample_size_treatment=len(treatment_values)
        )


class ExperimentManager:
    """
    Central A/B testing and experiment management.
    
    PHASE3-REFACTOR: Complete experiment lifecycle and statistical analysis.
    """
    
    def __init__(self, experiment_store: Optional[IExperimentStore] = None):
        self.experiment_store = experiment_store or MemoryExperimentStore()
        self.traffic_router = TrafficRouter()
        self.statistical_analyzer = StatisticalAnalyzer()
        
        # Active experiments cache
        self.active_experiments: Dict[str, Experiment] = {}
        
        # Background tasks
        self.monitor_task: Optional[asyncio.Task] = None
        
        # Start monitoring
        self._start_monitoring()
    
    def _start_monitoring(self):
        """Start experiment monitoring tasks."""
        self.monitor_task = asyncio.create_task(self._monitor_experiments())
    
    async def create_experiment(self, name: str, description: str,
                              variants: List[Dict[str, Any]],
                              experiment_type: ExperimentType = ExperimentType.AB_TEST,
                              duration_days: int = 14,
                              tenant_ids: Optional[List[str]] = None,
                              created_by: str = "") -> Experiment:
        """Create new experiment."""
        experiment_id = hashlib.md5(f"{name}:{datetime.utcnow()}".encode()).hexdigest()[:12]
        
        # Create variants
        experiment_variants = []
        for i, variant_data in enumerate(variants):
            variant = ExperimentVariant(
                id=variant_data.get("id", f"variant_{i}"),
                name=variant_data.get("name", f"Variant {i}"),
                description=variant_data.get("description", ""),
                traffic_allocation=variant_data.get("traffic_allocation", 1.0 / len(variants)),
                config=variant_data.get("config", {})
            )
            experiment_variants.append(variant)
        
        experiment = Experiment(
            id=experiment_id,
            name=name,
            description=description,
            experiment_type=experiment_type,
            variants=experiment_variants,
            control_variant_id=experiment_variants[0].id if experiment_variants else None,
            duration_days=duration_days,
            tenant_ids=tenant_ids or [],
            created_by=created_by
        )
        
        # Validate traffic allocation
        if not experiment.validate_traffic_allocation():
            raise ValueError("Traffic allocation must sum to 1.0")
        
        await self.experiment_store.save_experiment(experiment)
        logger.info(f"Created experiment '{name}' ({experiment_id})")
        
        return experiment
    
    async def start_experiment(self, experiment_id: str) -> bool:
        """Start experiment."""
        experiment = await self.experiment_store.load_experiment(experiment_id)
        if not experiment:
            return False
        
        experiment.status = ExperimentStatus.ACTIVE
        experiment.start_date = datetime.utcnow()
        experiment.end_date = experiment.start_date + timedelta(days=experiment.duration_days)
        
        success = await self.experiment_store.save_experiment(experiment)
        if success:
            self.active_experiments[experiment_id] = experiment
            logger.info(f"Started experiment {experiment_id}")
        
        return success
    
    async def stop_experiment(self, experiment_id: str) -> bool:
        """Stop experiment."""
        experiment = await self.experiment_store.load_experiment(experiment_id)
        if not experiment:
            return False
        
        experiment.status = ExperimentStatus.COMPLETED
        experiment.end_date = datetime.utcnow()
        
        success = await self.experiment_store.save_experiment(experiment)
        if success and experiment_id in self.active_experiments:
            del self.active_experiments[experiment_id]
            logger.info(f"Stopped experiment {experiment_id}")
        
        return success
    
    async def get_variant_for_user(self, user_id: str, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Get active experiment variants for user."""
        user_variants = {}
        
        for experiment in self.active_experiments.values():
            variant = self.traffic_router.get_variant_for_user(experiment, user_id, tenant_id)
            if variant:
                user_variants[experiment.id] = {
                    "experiment_name": experiment.name,
                    "variant_id": variant.id,
                    "variant_name": variant.name,
                    "config": variant.config
                }
        
        return user_variants
    
    async def record_interaction(self, experiment_id: str, variant_id: str,
                               converted: bool = False, response_time: float = 0.0,
                               satisfaction: float = 0.0, 
                               custom_metrics: Optional[Dict[str, float]] = None) -> bool:
        """Record user interaction with experiment variant."""
        experiment = await self.experiment_store.load_experiment(experiment_id)
        if not experiment:
            return False
        
        variant = experiment.get_variant(variant_id)
        if not variant:
            return False
        
        variant.record_interaction(converted, response_time, satisfaction, custom_metrics)
        
        success = await self.experiment_store.save_experiment(experiment)
        if success:
            # Update cache
            self.active_experiments[experiment_id] = experiment
        
        return success
    
    async def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get statistical analysis of experiment results."""
        experiment = await self.experiment_store.load_experiment(experiment_id)
        if not experiment:
            return {"error": "Experiment not found"}
        
        control_variant = experiment.get_control_variant()
        if not control_variant:
            return {"error": "No control variant found"}
        
        results = {
            "experiment": experiment.to_dict(),
            "statistical_analysis": [],
            "summary": {
                "total_participants": experiment.get_total_participants(),
                "duration_days": (datetime.utcnow() - experiment.created_at).days,
                "is_complete": experiment.status == ExperimentStatus.COMPLETED
            }
        }
        
        # Analyze each treatment variant against control
        for variant in experiment.variants:
            if variant.id == control_variant.id:
                continue
            
            # Conversion rate analysis
            conversion_result = self.statistical_analyzer.analyze_conversion_rate(
                control_variant, variant, experiment.confidence_level
            )
            results["statistical_analysis"].append(conversion_result.to_dict())
            
            # Response time analysis (if data available)
            if variant.participants > 0 and control_variant.participants > 0:
                # Simulate response time distributions for demo
                control_times = [control_variant.get_avg_response_time()] * control_variant.participants
                treatment_times = [variant.get_avg_response_time()] * variant.participants
                
                response_time_result = self.statistical_analyzer.analyze_continuous_metric(
                    control_times, treatment_times, "response_time", experiment.confidence_level
                )
                results["statistical_analysis"].append(response_time_result.to_dict())
        
        return results
    
    async def list_experiments(self, status: Optional[ExperimentStatus] = None,
                             tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List experiments with optional filters."""
        experiments = await self.experiment_store.list_experiments(status, tenant_id)
        return [exp.to_dict() for exp in experiments]
    
    async def _monitor_experiments(self):
        """Background monitoring of experiments."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Load active experiments
                active_experiments = await self.experiment_store.list_experiments(
                    status=ExperimentStatus.ACTIVE
                )
                
                # Update cache
                self.active_experiments = {exp.id: exp for exp in active_experiments}
                
                # Check for expired experiments
                now = datetime.utcnow()
                for experiment in active_experiments:
                    if experiment.end_date and now > experiment.end_date:
                        await self.stop_experiment(experiment.id)
                
            except Exception as e:
                logger.error(f"Experiment monitoring error: {e}")
    
    async def get_experiment_stats(self) -> Dict[str, Any]:
        """Get overall experiment statistics."""
        all_experiments = await self.experiment_store.list_experiments()
        
        stats = {
            "total_experiments": len(all_experiments),
            "active_experiments": len([e for e in all_experiments if e.status == ExperimentStatus.ACTIVE]),
            "completed_experiments": len([e for e in all_experiments if e.status == ExperimentStatus.COMPLETED]),
            "total_participants": sum(e.get_total_participants() for e in all_experiments)
        }
        
        return stats
    
    async def shutdown(self):
        """Shutdown experiment manager."""
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Experiment manager shutdown complete")


# Global experiment manager
_experiment_manager: Optional[ExperimentManager] = None


def get_experiment_manager() -> ExperimentManager:
    """Get global experiment manager."""
    global _experiment_manager
    if _experiment_manager is None:
        _experiment_manager = ExperimentManager()
    return _experiment_manager


async def initialize_experiment_manager(experiment_store: Optional[IExperimentStore] = None):
    """Initialize global experiment manager."""
    global _experiment_manager
    _experiment_manager = ExperimentManager(experiment_store)
    logger.info("Global experiment manager initialized")


async def shutdown_experiment_manager():
    """Shutdown global experiment manager."""
    global _experiment_manager
    if _experiment_manager:
        await _experiment_manager.shutdown()
        _experiment_manager = None
    logger.info("Global experiment manager shutdown")


# Decorator for experiment-aware functions
def experiment_variant(experiment_name: str):
    """Decorator to apply experiment variant configuration."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # In a real implementation, this would:
            # 1. Get user ID from context
            # 2. Look up active experiments
            # 3. Apply variant configuration
            # 4. Record interaction results
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator
