# Name:  Joshua Jerry Selorm Yegbe

# Index Number: 10211100403


import pandas as pd
import re
import json
import os
from datetime import datetime

class RagEvaluator:
    def __init__(self):
        """Initialize the evaluator"""
        self.eval_results = []
        self.gpt_comparison = []
        
    def evaluate_response(self, question, context, response, ground_truth=None):
        """Evaluate a single RAG response"""
        # Basic evaluation metrics
        eval_result = {
            "question": question,
            "response": response,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": {}
        }
        
        # Context relevance - simple keyword matching
        keywords = self._extract_keywords(question)
        context_relevance = sum(1 for kw in keywords if kw.lower() in context.lower()) / max(1, len(keywords))
        eval_result["metrics"]["context_relevance"] = context_relevance
        
        # Response completeness - does it address the question keywords?
        response_completeness = sum(1 for kw in keywords if kw.lower() in response.lower()) / max(1, len(keywords))
        eval_result["metrics"]["response_completeness"] = response_completeness
        
        # Response conciseness - ratio of response length to context length
        response_conciseness = min(1.0, len(context) / max(1, len(response)))
        eval_result["metrics"]["response_conciseness"] = response_conciseness
        
        # Ground truth comparison if available
        if ground_truth:
            eval_result["ground_truth"] = ground_truth
            # Simple overlap score
            response_words = set(re.findall(r'\b\w+\b', response.lower()))
            truth_words = set(re.findall(r'\b\w+\b', ground_truth.lower()))
            if truth_words:
                overlap = len(response_words.intersection(truth_words)) / len(truth_words)
                eval_result["metrics"]["ground_truth_overlap"] = overlap
        
        # Store the evaluation
        self.eval_results.append(eval_result)
        return eval_result
    
    def compare_with_chatgpt(self, question, rag_response, chatgpt_response):
        """Compare RAG response with ChatGPT response"""
        comparison = {
            "question": question,
            "rag_response": rag_response,
            "chatgpt_response": chatgpt_response,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "comparison": {}
        }
        
        # Calculate similarity between responses
        rag_words = set(re.findall(r'\b\w+\b', rag_response.lower()))
        gpt_words = set(re.findall(r'\b\w+\b', chatgpt_response.lower()))
        
        # Jaccard similarity
        union = len(rag_words.union(gpt_words))
        intersection = len(rag_words.intersection(gpt_words))
        similarity = intersection / max(1, union)
        comparison["comparison"]["response_similarity"] = similarity
        
        # Length comparison
        rag_length = len(rag_response.split())
        gpt_length = len(chatgpt_response.split())
        length_ratio = min(rag_length, gpt_length) / max(1, max(rag_length, gpt_length))
        comparison["comparison"]["length_ratio"] = length_ratio
        
        # Store the comparison
        self.gpt_comparison.append(comparison)
        return comparison
    
    def _extract_keywords(self, text):
        """Extract potential keywords from the question"""
        # Remove stopwords and punctuation
        stopwords = ["a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "of"]
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        return keywords
    
    def save_evaluations(self, directory="./evaluations"):
        """Save evaluation results to JSON files"""
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # Save RAG evaluations
        if self.eval_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f"{directory}/rag_eval_{timestamp}.json", 'w') as f:
                json.dump(self.eval_results, f, indent=2)
        
        # Save GPT comparisons
        if self.gpt_comparison:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f"{directory}/gpt_comparison_{timestamp}.json", 'w') as f:
                json.dump(self.gpt_comparison, f, indent=2)
    
    def generate_summary(self):
        """Generate a summary of all evaluations"""
        summary = {
            "total_evaluations": len(self.eval_results),
            "total_comparisons": len(self.gpt_comparison),
            "average_metrics": {},
            "comparison_averages": {}
        }
        
        # Calculate average RAG metrics
        if self.eval_results:
            metrics_keys = set()
            for result in self.eval_results:
                metrics_keys.update(result["metrics"].keys())
                
            for key in metrics_keys:
                values = [r["metrics"].get(key, 0) for r in self.eval_results if key in r["metrics"]]
                if values:
                    summary["average_metrics"][key] = sum(values) / len(values)
        
        # Calculate average comparison metrics
        if self.gpt_comparison:
            comparison_keys = set()
            for comp in self.gpt_comparison:
                comparison_keys.update(comp["comparison"].keys())
                
            for key in comparison_keys:
                values = [c["comparison"].get(key, 0) for c in self.gpt_comparison if key in c["comparison"]]
                if values:
                    summary["comparison_averages"][key] = sum(values) / len(values)
        
        return summary
