"""
API Example and Testing Script for Claims Risk Classification

This script demonstrates how to interact with the Claims Risk Classification API
including authentication, single predictions, batch processing, and monitoring.
"""

import asyncio
import httpx
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APIClient:
    """
    Client for interacting with Claims Risk Classification API
    
    Features:
    - Authentication management
    - Single and batch predictions
    - Model management
    - Health monitoring
    - Error handling and retries
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = None):
        """
        Initialize API client
        
        Args:
            base_url: Base URL of the API
            api_key: API key for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = None
        
        # Default headers
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = httpx.AsyncClient(timeout=30.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.aclose()
    
    async def _make_request(self, method: str, endpoint: str, 
                          data: Dict = None, params: Dict = None) -> Dict[str, Any]:
        """
        Make HTTP request to API
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request data
            params: Query parameters
            
        Returns:
            Response data
            
        Raises:
            Exception: If request fails
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = await self.session.get(url, headers=self.headers, params=params)
            elif method.upper() == "POST":
                response = await self.session.post(url, headers=self.headers, json=data)
            elif method.upper() == "PUT":
                response = await self.session.put(url, headers=self.headers, json=data)
            elif method.upper() == "DELETE":
                response = await self.session.delete(url, headers=self.headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            error_detail = "Unknown error"
            try:
                error_response = e.response.json()
                error_detail = error_response.get("message", str(e))
            except Exception:
                error_detail = str(e)
            
            logger.error(f"HTTP {e.response.status_code}: {error_detail}")
            raise Exception(f"API request failed: {error_detail}")
        
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        return await self._make_request("GET", "/health")
    
    async def detailed_health_check(self) -> Dict[str, Any]:
        """Get detailed health information"""
        return await self._make_request("GET", "/health/detailed")
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models"""
        return await self._make_request("GET", "/models")
    
    async def load_model(self, model_name: str) -> Dict[str, Any]:
        """Load a specific model"""
        return await self._make_request("POST", f"/models/{model_name}/load")
    
    async def predict_single(self, claim_data: Dict[str, Any], 
                           model_name: str = None) -> Dict[str, Any]:
        """
        Make single claim prediction
        
        Args:
            claim_data: Claim data for prediction
            model_name: Optional model name to use
            
        Returns:
            Prediction response
        """
        endpoint = "/predict"
        if model_name:
            endpoint += f"?model_name={model_name}"
        
        return await self._make_request("POST", endpoint, claim_data)
    
    async def predict_batch(self, claims_data: List[Dict[str, Any]], 
                          batch_id: str = None, model_name: str = None) -> Dict[str, Any]:
        """
        Make batch predictions
        
        Args:
            claims_data: List of claims for prediction
            batch_id: Optional batch identifier
            model_name: Optional model name to use
            
        Returns:
            Batch prediction response
        """
        endpoint = "/predict/batch"
        params = {}
        if model_name:
            params["model_name"] = model_name
        
        payload = {
            "batch_id": batch_id,
            "claims": claims_data
        }
        
        return await self._make_request("POST", endpoint, payload)
    
    async def get_drift_status(self, model_name: str) -> Dict[str, Any]:
        """Get data drift status for a model"""
        return await self._make_request("GET", f"/monitoring/drift?model_name={model_name}")
    
    async def get_performance_metrics(self, model_name: str, days: int = 7) -> Dict[str, Any]:
        """Get performance metrics for a model"""
        params = {"model_name": model_name, "days": days}
        return await self._make_request("GET", "/monitoring/performance", params=params)


def create_sample_claim_data() -> Dict[str, Any]:
    """Create sample claim data for testing"""
    return {
        "claim_id": f"TEST_CLAIM_{int(time.time())}",
        "claim_data": {
            "claim_amount": 15000.0,
            "claimant_age": 35,
            "policy_duration_months": 24,
            "previous_claims": 1,
            "region": "North",
            "claim_type": "Auto",
            "day_of_week": "Monday",
            "weather_condition": "Clear",
            "claim_description": "Vehicle collision on interstate highway resulting in significant front-end damage and airbag deployment",
            "policy_premium": 1200.0,
            "deductible_amount": 500.0
        },
        "return_probabilities": True,
        "return_feature_importance": True,
        "return_business_impact": True
    }


def create_batch_claim_data(size: int = 5) -> List[Dict[str, Any]]:
    """Create batch of sample claim data"""
    base_claim = create_sample_claim_data()
    claims = []
    
    # Generate variations
    variations = [
        {"claim_amount": 5000.0, "claimant_age": 25, "previous_claims": 0},
        {"claim_amount": 25000.0, "claimant_age": 45, "previous_claims": 2},
        {"claim_amount": 8000.0, "claimant_age": 60, "previous_claims": 1},
        {"claim_amount": 35000.0, "claimant_age": 30, "previous_claims": 3},
        {"claim_amount": 12000.0, "claimant_age": 50, "previous_claims": 0}
    ]
    
    for i in range(size):
        claim = base_claim.copy()
        claim["claim_id"] = f"BATCH_CLAIM_{i+1}_{int(time.time())}"
        
        # Apply variation
        if i < len(variations):
            claim["claim_data"].update(variations[i])
        
        claims.append(claim)
    
    return claims


async def test_api_functionality():
    """Comprehensive API testing"""
    logger.info("Starting comprehensive API testing...")
    
    # Configuration
    api_key = "test_api_key_12345"  # In real use, this should be secure
    
    async with APIClient(api_key=api_key) as client:
        
        try:
            # 1. Health Check
            logger.info("=" * 50)
            logger.info("1. HEALTH CHECK")
            logger.info("=" * 50)
            
            health = await client.health_check()
            logger.info(f"Health Status: {health.get('status')}")
            logger.info(f"Available Models: {health.get('available_models', [])}")
            
            # Detailed health check (requires authentication)
            try:
                detailed_health = await client.detailed_health_check()
                logger.info(f"System Status: {detailed_health.get('api_status')}")
            except Exception as e:
                logger.warning(f"Detailed health check failed (expected without valid API key): {e}")
            
            # 2. Model Management
            logger.info("\\n" + "=" * 50)
            logger.info("2. MODEL MANAGEMENT")
            logger.info("=" * 50)
            
            try:
                models = await client.list_models()
                logger.info(f"Available Models: {len(models)}\")\n                for model in models:\n                    logger.info(f\"  - {model.get('model_name')}: {model.get('status')}\")\n            except Exception as e:\n                logger.warning(f\"Model listing failed (expected without valid API key): {e}\")\n            \n            # 3. Single Prediction\n            logger.info(\"\\n\" + \"=\" * 50)\n            logger.info(\"3. SINGLE PREDICTION\")\n            logger.info(\"=\" * 50)\n            \n            sample_claim = create_sample_claim_data()\n            logger.info(f\"Testing claim ID: {sample_claim['claim_id']}\")\n            logger.info(f\"Claim amount: ${sample_claim['claim_data']['claim_amount']:,.2f}\")\n            \n            try:\n                start_time = time.time()\n                prediction = await client.predict_single(sample_claim)\n                prediction_time = (time.time() - start_time) * 1000\n                \n                logger.info(f\"Prediction Result:\")\n                logger.info(f\"  Risk Level: {prediction.get('predicted_risk')}\")\n                logger.info(f\"  Confidence: {prediction.get('confidence_score', 0):.3f}\")\n                logger.info(f\"  High Risk Probability: {prediction.get('probability_high_risk', 0):.3f}\")\n                logger.info(f\"  Processing Time: {prediction_time:.1f}ms\")\n                \n                # Business impact\n                if 'business_impact' in prediction:\n                    bi = prediction['business_impact']\n                    logger.info(f\"  Estimated Cost: ${bi.get('estimated_cost', 0):,.2f}\")\n                    logger.info(f\"  Potential Savings: ${bi.get('potential_savings', 0):,.2f}\")\n                    logger.info(f\"  Recommendation: {bi.get('recommendation')}\")\n                \n                # Feature importance\n                if 'feature_importance' in prediction and prediction['feature_importance']:\n                    logger.info(f\"  Top Important Features:\")\n                    for feat in prediction['feature_importance'][:3]:\n                        logger.info(f\"    - {feat.get('feature')}: {feat.get('importance', 0):.3f}\")\n                \n            except Exception as e:\n                logger.error(f\"Single prediction failed: {e}\")\n            \n            # 4. Batch Prediction\n            logger.info(\"\\n\" + \"=\" * 50)\n            logger.info(\"4. BATCH PREDICTION\")\n            logger.info(\"=\" * 50)\n            \n            batch_claims = create_batch_claim_data(3)\n            batch_id = f\"TEST_BATCH_{int(time.time())}\"\n            logger.info(f\"Testing batch ID: {batch_id}\")\n            logger.info(f\"Batch size: {len(batch_claims)} claims\")\n            \n            try:\n                start_time = time.time()\n                batch_result = await client.predict_batch(batch_claims, batch_id)\n                batch_time = (time.time() - start_time) * 1000\n                \n                logger.info(f\"Batch Processing Result:\")\n                logger.info(f\"  Processed Claims: {len(batch_result.get('predictions', []))}\")\n                logger.info(f\"  Processing Time: {batch_time:.1f}ms\")\n                \n                # Batch statistics\n                if 'batch_statistics' in batch_result:\n                    stats = batch_result['batch_statistics']\n                    logger.info(f\"  High Risk Claims: {stats.get('high_risk_claims', 0)}\")\n                    logger.info(f\"  Low Risk Claims: {stats.get('low_risk_claims', 0)}\")\n                    logger.info(f\"  High Risk Percentage: {stats.get('high_risk_percentage', 0):.1f}%\")\n                    logger.info(f\"  Average Confidence: {stats.get('average_confidence', 0):.3f}\")\n                \n                # Individual predictions summary\n                logger.info(f\"  Individual Results:\")\n                for i, pred in enumerate(batch_result.get('predictions', [])[:5]):\n                    logger.info(\n                        f\"    {i+1}. {pred.get('claim_id')}: {pred.get('predicted_risk')} \"\n                        f\"(conf: {pred.get('confidence_score', 0):.3f})\"\n                    )\n                \n            except Exception as e:\n                logger.error(f\"Batch prediction failed: {e}\")\n            \n            # 5. Monitoring (if available)\n            logger.info(\"\\n\" + \"=\" * 50)\n            logger.info(\"5. MONITORING\")\n            logger.info(\"=\" * 50)\n            \n            try:\n                # Try to get drift status\n                drift_status = await client.get_drift_status(\"Random Forest\")\n                logger.info(f\"Drift Status: {drift_status.get('drift_detected')}\")\n                logger.info(f\"Drift Score: {drift_status.get('drift_score', 0):.4f}\")\n            except Exception as e:\n                logger.warning(f\"Drift monitoring not available: {e}\")\n            \n            try:\n                # Try to get performance metrics\n                perf_metrics = await client.get_performance_metrics(\"Random Forest\")\n                logger.info(f\"Performance Trend: {perf_metrics.get('trend')}\")\n                if 'metrics' in perf_metrics:\n                    metrics = perf_metrics['metrics']\n                    logger.info(f\"Recent Accuracy: {metrics.get('accuracy', 0):.3f}\")\n                    logger.info(f\"Recent F1-Score: {metrics.get('f1_score', 0):.3f}\")\n            except Exception as e:\n                logger.warning(f\"Performance monitoring not available: {e}\")\n            \n            logger.info(\"\\n\" + \"=\" * 50)\n            logger.info(\"API TESTING COMPLETED SUCCESSFULLY\")\n            logger.info(\"=\" * 50)\n            \n        except Exception as e:\n            logger.error(f\"API testing failed: {e}\")\n            raise\n\n\nasync def performance_test():\n    \"\"\"Performance testing with multiple concurrent requests\"\"\"\n    logger.info(\"Starting performance testing...\")\n    \n    api_key = \"test_api_key_12345\"\n    \n    async with APIClient(api_key=api_key) as client:\n        \n        # Test concurrent single predictions\n        logger.info(\"Testing concurrent single predictions...\")\n        \n        async def make_prediction(i):\n            claim = create_sample_claim_data()\n            claim['claim_id'] = f\"PERF_TEST_{i}_{int(time.time())}\"\n            \n            start_time = time.time()\n            try:\n                result = await client.predict_single(claim)\n                duration = (time.time() - start_time) * 1000\n                return {'success': True, 'duration': duration, 'result': result}\n            except Exception as e:\n                duration = (time.time() - start_time) * 1000\n                return {'success': False, 'duration': duration, 'error': str(e)}\n        \n        # Run concurrent requests\n        num_requests = 10\n        start_time = time.time()\n        \n        tasks = [make_prediction(i) for i in range(num_requests)]\n        results = await asyncio.gather(*tasks)\n        \n        total_time = (time.time() - start_time) * 1000\n        \n        # Analyze results\n        successful = [r for r in results if r['success']]\n        failed = [r for r in results if not r['success']]\n        \n        logger.info(f\"Performance Test Results:\")\n        logger.info(f\"  Total Requests: {num_requests}\")\n        logger.info(f\"  Successful: {len(successful)}\")\n        logger.info(f\"  Failed: {len(failed)}\")\n        logger.info(f\"  Success Rate: {len(successful)/num_requests*100:.1f}%\")\n        logger.info(f\"  Total Time: {total_time:.1f}ms\")\n        logger.info(f\"  Requests/sec: {num_requests/(total_time/1000):.1f}\")\n        \n        if successful:\n            durations = [r['duration'] for r in successful]\n            logger.info(f\"  Avg Response Time: {sum(durations)/len(durations):.1f}ms\")\n            logger.info(f\"  Min Response Time: {min(durations):.1f}ms\")\n            logger.info(f\"  Max Response Time: {max(durations):.1f}ms\")\n        \n        if failed:\n            logger.info(f\"  Errors:\")\n            for i, result in enumerate(failed[:3]):\n                logger.info(f\"    {i+1}. {result['error']}\")\n\n\ndef create_csv_test_data(filename: str = \"test_claims.csv\", num_claims: int = 50):\n    \"\"\"\n    Create CSV file with test claims data\n    \n    Args:\n        filename: Output filename\n        num_claims: Number of claims to generate\n    \"\"\"\n    logger.info(f\"Creating test data CSV: {filename}\")\n    \n    claims = create_batch_claim_data(num_claims)\n    \n    # Flatten claim data for CSV\n    csv_data = []\n    for claim in claims:\n        row = claim['claim_data'].copy()\n        row['claim_id'] = claim['claim_id']\n        csv_data.append(row)\n    \n    df = pd.DataFrame(csv_data)\n    df.to_csv(filename, index=False)\n    \n    logger.info(f\"Created {filename} with {len(df)} claims\")\n    logger.info(f\"Columns: {list(df.columns)}\")\n    \n    return filename\n\n\ndef print_usage_examples():\n    \"\"\"Print usage examples for the API\"\"\"\n    print(\"\"\"\n    CLAIMS RISK CLASSIFICATION API - USAGE EXAMPLES\n    ===============================================\n    \n    1. Start the API server:\n       python src/api/server.py --dev\n    \n    2. Run comprehensive tests:\n       python examples/api_example.py test\n    \n    3. Run performance tests:\n       python examples/api_example.py performance\n    \n    4. Create test data:\n       python examples/api_example.py create-data\n    \n    5. Using curl for single prediction:\n    \n    curl -X POST \"http://localhost:8000/predict\" \\\n         -H \"Authorization: Bearer your_api_key\" \\\n         -H \"Content-Type: application/json\" \\\n         -d '{\n           \"claim_id\": \"TEST_001\",\n           \"claim_data\": {\n             \"claim_amount\": 15000.0,\n             \"claimant_age\": 35,\n             \"policy_duration_months\": 24,\n             \"previous_claims\": 1,\n             \"region\": \"North\",\n             \"claim_type\": \"Auto\",\n             \"day_of_week\": \"Monday\",\n             \"weather_condition\": \"Clear\",\n             \"claim_description\": \"Vehicle collision on highway\"\n           }\n         }'\n    \n    6. Using Python requests:\n    \n    import requests\n    \n    response = requests.post(\n        \"http://localhost:8000/predict\",\n        headers={\"Authorization\": \"Bearer your_api_key\"},\n        json={\n            \"claim_id\": \"TEST_001\",\n            \"claim_data\": {\n                \"claim_amount\": 15000.0,\n                \"claimant_age\": 35,\n                # ... other fields\n            }\n        }\n    )\n    \n    result = response.json()\n    print(f\"Risk Level: {result['predicted_risk']}\")\n    print(f\"Confidence: {result['confidence_score']}\")\n    \n    7. Health check:\n       curl http://localhost:8000/health\n    \n    8. API Documentation:\n       http://localhost:8000/docs\n    \"\"\")\n\n\nasync def main():\n    \"\"\"Main function to handle different test modes\"\"\"\n    import sys\n    \n    if len(sys.argv) < 2:\n        print_usage_examples()\n        return\n    \n    command = sys.argv[1].lower()\n    \n    if command == \"test\":\n        await test_api_functionality()\n    elif command == \"performance\":\n        await performance_test()\n    elif command == \"create-data\":\n        create_csv_test_data()\n    elif command == \"usage\" or command == \"help\":\n        print_usage_examples()\n    else:\n        logger.error(f\"Unknown command: {command}\")\n        print_usage_examples()\n\n\nif __name__ == \"__main__\":\n    asyncio.run(main())"