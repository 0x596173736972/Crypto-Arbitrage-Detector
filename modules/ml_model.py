import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MLPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_model_trained = False
        self.feature_importance = {}
        self.model_metrics = {}

    def create_features(self, opportunities_data: pd.DataFrame) -> pd.DataFrame:
        """Create features for machine learning model."""
        try:
            if opportunities_data.empty:
                return pd.DataFrame()
            
            # Make a copy to avoid modifying original data
            df = opportunities_data.copy()
            
            # Basic features
            features = pd.DataFrame()
            features['spread_pct'] = df['spread_pct']
            features['risk_score'] = df['risk_score']
            features['opportunity_score'] = df['opportunity_score']
            
            # Profit-related features
            if 'net_profit' in df.columns:
                features['net_profit'] = df['net_profit']
                features['profit_margin'] = df['net_profit'] / df.get('trade_amount', 1000)
            
            # Price-related features
            if 'buy_price' in df.columns and 'sell_price' in df.columns:
                features['price_ratio'] = df['sell_price'] / df['buy_price']
                features['avg_price'] = (df['buy_price'] + df['sell_price']) / 2
                features['price_volatility'] = abs(df['sell_price'] - df['buy_price']) / features['avg_price']
            
            # Exchange reliability features
            exchange_reliability = {
                'binance': 0.95,
                'coinbase': 0.90,
                'kraken': 0.88
            }
            
            if 'buy_exchange' in df.columns and 'sell_exchange' in df.columns:
                features['buy_exchange_reliability'] = df['buy_exchange'].map(exchange_reliability).fillna(0.5)
                features['sell_exchange_reliability'] = df['sell_exchange'].map(exchange_reliability).fillna(0.5)
                features['avg_exchange_reliability'] = (features['buy_exchange_reliability'] + 
                                                      features['sell_exchange_reliability']) / 2
            
            # Time-based features (if timestamp available)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                features['hour'] = df['timestamp'].dt.hour
                features['day_of_week'] = df['timestamp'].dt.dayofweek
                features['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)
                
                # Market session features
                features['is_us_trading_hours'] = ((df['timestamp'].dt.hour >= 9) & 
                                                 (df['timestamp'].dt.hour <= 16)).astype(int)
                features['is_asian_trading_hours'] = ((df['timestamp'].dt.hour >= 22) | 
                                                    (df['timestamp'].dt.hour <= 6)).astype(int)
            
            # Rolling statistical features (if enough data)
            if len(df) > 10:
                # Rolling averages
                features['spread_ma_5'] = df['spread_pct'].rolling(5, min_periods=1).mean()
                features['spread_ma_10'] = df['spread_pct'].rolling(10, min_periods=1).mean()
                
                # Rolling volatility
                features['spread_volatility'] = df['spread_pct'].rolling(5, min_periods=1).std().fillna(0)
                
                # Z-score features
                rolling_mean = df['spread_pct'].rolling(10, min_periods=1).mean()
                rolling_std = df['spread_pct'].rolling(10, min_periods=1).std()
                features['spread_zscore'] = ((df['spread_pct'] - rolling_mean) / 
                                            rolling_std.replace(0, 1)).fillna(0)
            
            # Crypto-specific features
            if 'crypto' in df.columns:
                # One-hot encode crypto types
                crypto_dummies = pd.get_dummies(df['crypto'], prefix='crypto')
                features = pd.concat([features, crypto_dummies], axis=1)
            
            # Fill any remaining NaN values
            features = features.fillna(0)
            
            # Ensure we have the same features during prediction
            self.feature_columns = features.columns.tolist()
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating features: {str(e)}")
            return pd.DataFrame()

    def create_target(self, opportunities_data: pd.DataFrame) -> pd.Series:
        """Create target variable for supervised learning."""
        try:
            # Define profitability based on multiple criteria
            target = pd.Series(0, index=opportunities_data.index)
            
            # Criteria for a "good" opportunity
            profitable = (
                (opportunities_data['spread_pct'] >= 0.5) &  # Minimum spread
                (opportunities_data['risk_score'] <= 6) &    # Low risk
                (opportunities_data['opportunity_score'] >= 5.0) &  # Good score
                (opportunities_data.get('net_profit', 0) > 0)  # Positive profit
            )
            
            target[profitable] = 1
            
            return target
            
        except Exception as e:
            logger.error(f"Error creating target: {str(e)}")
            return pd.Series(0, index=opportunities_data.index)

    def train_model(self, historical_data: pd.DataFrame) -> float:
        """Train the machine learning model."""
        try:
            if len(historical_data) < 50:
                raise ValueError("Need at least 50 historical records for training")
            
            # Create features and target
            X = self.create_features(historical_data)
            y = self.create_target(historical_data)
            
            if X.empty or len(X) != len(y):
                raise ValueError("Feature creation failed or size mismatch")
            
            # Check for class imbalance
            class_distribution = y.value_counts()
            logger.info(f"Class distribution: {class_distribution.to_dict()}")
            
            if len(class_distribution) == 1:
                # If only one class, create a balanced dataset
                logger.warning("Only one class found, creating synthetic examples")
                if class_distribution.index[0] == 0:
                    # Add some positive examples
                    positive_indices = X.sample(min(10, len(X)//2)).index
                    y.loc[positive_indices] = 1
                else:
                    # Add some negative examples
                    negative_indices = X.sample(min(10, len(X)//2)).index
                    y.loc[negative_indices] = 0
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train the model
            self.model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
            
            # Feature importance
            self.feature_importance = dict(zip(X.columns, self.model.feature_importances_))
            
            # Store model metrics
            self.model_metrics = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_count': len(X.columns),
                'positive_class_ratio': y.sum() / len(y)
            }
            
            self.is_model_trained = True
            
            logger.info(f"Model trained successfully. Accuracy: {accuracy:.4f}")
            logger.info(f"Cross-validation score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            self.is_model_trained = False
            return 0.0

    def predict_opportunities(self, opportunities: List[Dict]) -> List[int]:
        """Predict if opportunities are profitable."""
        try:
            if not self.is_model_trained:
                logger.warning("Model not trained, returning default predictions")
                return [0] * len(opportunities)
            
            # Convert to DataFrame
            df = pd.DataFrame(opportunities)
            
            # Create features
            X = self.create_features(df)
            
            # Ensure we have the same features as training
            for col in self.feature_columns:
                if col not in X.columns:
                    X[col] = 0
            
            X = X[self.feature_columns]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            predictions = self.model.predict(X_scaled)
            
            return predictions.tolist()
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return [0] * len(opportunities)

    def predict_probability(self, opportunities: List[Dict]) -> List[float]:
        """Predict probability of profitable opportunities."""
        try:
            if not self.is_model_trained:
                return [0.5] * len(opportunities)
            
            # Convert to DataFrame
            df = pd.DataFrame(opportunities)
            
            # Create features
            X = self.create_features(df)
            
            # Ensure we have the same features as training
            for col in self.feature_columns:
                if col not in X.columns:
                    X[col] = 0
            
            X = X[self.feature_columns]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Get probability of positive class
            probabilities = self.model.predict_proba(X_scaled)[:, 1]
            
            return probabilities.tolist()
            
        except Exception as e:
            logger.error(f"Error predicting probabilities: {str(e)}")
            return [0.5] * len(opportunities)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if not self.is_model_trained:
            return {}
        return self.feature_importance

    def get_model_metrics(self) -> Dict:
        """Get model performance metrics."""
        return self.model_metrics

    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self.is_model_trained

    def save_model(self, filepath: str):
        """Save trained model to file."""
        try:
            if not self.is_model_trained:
                raise ValueError("Model not trained")
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'feature_importance': self.feature_importance,
                'model_metrics': self.model_metrics
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")

    def load_model(self, filepath: str):
        """Load trained model from file."""
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.feature_importance = model_data['feature_importance']
            self.model_metrics = model_data['model_metrics']
            self.is_model_trained = True
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.is_model_trained = False

    def retrain_with_new_data(self, new_data: pd.DataFrame):
        """Retrain model with new data (incremental learning simulation)."""
        try:
            if not self.is_model_trained:
                return self.train_model(new_data)
            
            # For Random Forest, we need to retrain the entire model
            # In a production environment, you might use online learning algorithms
            logger.info("Retraining model with new data...")
            return self.train_model(new_data)
            
        except Exception as e:
            logger.error(f"Error retraining model: {str(e)}")
            return 0.0

    def analyze_prediction_confidence(self, opportunities: List[Dict]) -> List[Dict]:
        """Analyze prediction confidence and provide explanations."""
        try:
            if not self.is_model_trained:
                return []
            
            predictions = self.predict_opportunities(opportunities)
            probabilities = self.predict_probability(opportunities)
            
            results = []
            for i, (opp, pred, prob) in enumerate(zip(opportunities, predictions, probabilities)):
                confidence = abs(prob - 0.5) * 2  # Convert to 0-1 confidence scale
                
                analysis = {
                    'prediction': pred,
                    'probability': prob,
                    'confidence': confidence,
                    'confidence_level': 'High' if confidence > 0.7 else 'Medium' if confidence > 0.4 else 'Low',
                    'explanation': self._generate_explanation(opp, pred, prob)
                }
                
                results.append(analysis)
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing prediction confidence: {str(e)}")
            return []

    def _generate_explanation(self, opportunity: Dict, prediction: int, probability: float) -> str:
        """Generate human-readable explanation for prediction."""
        try:
            explanation_parts = []
            
            # Spread analysis
            spread = opportunity.get('spread_pct', 0)
            if spread > 1.0:
                explanation_parts.append(f"High spread ({spread:.2f}%)")
            elif spread > 0.5:
                explanation_parts.append(f"Moderate spread ({spread:.2f}%)")
            else:
                explanation_parts.append(f"Low spread ({spread:.2f}%)")
            
            # Risk analysis
            risk_score = opportunity.get('risk_score', 5)
            if risk_score <= 3:
                explanation_parts.append("low risk")
            elif risk_score <= 6:
                explanation_parts.append("moderate risk")
            else:
                explanation_parts.append("high risk")
            
            # Opportunity score
            opp_score = opportunity.get('opportunity_score', 5)
            if opp_score >= 7:
                explanation_parts.append("strong opportunity score")
            elif opp_score >= 5:
                explanation_parts.append("decent opportunity score")
            else:
                explanation_parts.append("weak opportunity score")
            
            # Combine explanation
            base_explanation = ", ".join(explanation_parts)
            
            if prediction == 1:
                return f"Predicted as PROFITABLE ({probability:.1%} confidence): {base_explanation}"
            else:
                return f"Predicted as NOT PROFITABLE ({1-probability:.1%} confidence): {base_explanation}"
                
        except Exception as e:
            return f"Prediction: {'Profitable' if prediction == 1 else 'Not Profitable'} ({probability:.1%} confidence)"
