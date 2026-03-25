from app.config.config import Config
from app.common.logger import MedicalRAGLogger
from app.components.retriever import MedicalRAGRetriever

from ragas import evaluate
from ragas.metrics._faithfulness import faithfulness
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision
from ragas.metrics._context_recall import context_recall
from ragas.llms.base import LangchainLLMWrapper
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

from datasets import Dataset
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
import json
from datetime import datetime
import os

class ComprehensiveRAGEvaluator:
    """
    Complete RAG evaluation combining:
    1. Custom Retrieval Metrics (Precision@K, Recall@K, MRR, F1)
    2. RAGAS Generation Metrics (Faithfulness, Relevancy, etc.)
    """

    def __init__(self):
        """Initialize comprehensive evaluator"""
        self.logger = MedicalRAGLogger(__name__)
        self.retriever = MedicalRAGRetriever()

        self.logger.logger.info("Comprehensive RAG Evaluator initialized")
        self.logger.logger.info("   Retrieval Metrics: Precision@K, Recall@K, MRR, F1")
        self.logger.logger.info("   Generation Metrics: RAGAS framework")

    # PART 1: RETRIEVAL METRICS

    def evaluate_retrieval_metrics(self, test_cases: List[Dict], k_values: List[int] = [3, 5, 10]) -> Dict:
        """
        Evaluate retrieval quality with multiple metrics

        Args:
            test_cases: List of test cases with ground truth
            [
                {
                    "question": "What is diabetes?",
                    "relevant_doc_ids": ["vec_234", "vec_256"],  # Ground truth
                    "category": "endocrine"
                },
                ...
            ]
            k_values: List of K values for Precision@K and Recall@K

        Returns:
            Dictionary with retrieval metrics
        """
        try:
            self.logger.logger.info("=" * 60)
            self.logger.logger.info("EVALUATING RETRIEVAL METRICS")
            self.logger.logger.info("=" * 60)

            all_metrics = {}

            for k in k_values:
                self.logger.logger.info(f"\nEvaluating with K={k}...")

                precision_scores = []
                recall_scores = []
                mrr_scores = []
                ndcg_scores = []

                for i, test_case in enumerate(test_cases):
                    question = test_case['question']
                    relevant_ids = set(test_case.get('relevant_doc_ids', []))

                    if not relevant_ids:
                        self.logger.logger.warning(f"Skipping case {i + 1}: No ground truth")
                        continue

                        # Retrieve documents
                    results = self.retriever.retrieve_documents(
                        query = question,
                        top_k = k,
                        filters = test_case.get('filters')
                    )

                    retrieved_ids = [doc['id'] for doc in results]

                    # Calculate metrics
                    precision = self._calculate_precision_at_k(retrieved_ids, relevant_ids)
                    recall = self._calculate_recall_at_k(retrieved_ids, relevant_ids)
                    mrr = self._calculate_mrr(retrieved_ids, relevant_ids)
                    ndcg = self._calculate_ndcg(results, relevant_ids)

                    precision_scores.append(precision)
                    recall_scores.append(recall)
                    mrr_scores.append(mrr)
                    ndcg_scores.append(ndcg)

                    if (i + 1) % 10 == 0:
                        self.logger.logger.info(f"  Progress: {i + 1}/{len(test_cases)}")

                # Calculate aggregated metrics
                avg_precision = np.mean(precision_scores)
                avg_recall = np.mean(recall_scores)
                avg_mrr = np.mean(mrr_scores)
                avg_ndcg = np.mean(ndcg_scores)
                f1_score = self._calculate_f1(avg_precision, avg_recall)

                all_metrics[f'k_{k}'] = {
                    'precision@k': float(avg_precision),
                    'recall@k': float(avg_recall),
                    'mrr': float(avg_mrr),
                    'ndcg@k': float(avg_ndcg),
                    'f1_score': float(f1_score),
                    'num_queries': len(precision_scores)
                }

                self.logger.logger.info(f"\n  Results for K={k}:")
                self.logger.logger.info(f"    Precision@{k}: {avg_precision:.4f}")
                self.logger.logger.info(f"    Recall@{k}: {avg_recall:.4f}")
                self.logger.logger.info(f"    MRR: {avg_mrr:.4f}")
                self.logger.logger.info(f"    NDCG@{k}: {avg_ndcg:.4f}")
                self.logger.logger.info(f"    F1 Score: {f1_score:.4f}")
            
            self.logger.logger.info("\n Retrieval metrics evaluation completed")
            
            return all_metrics
        
        except Exception as e:
            self.logger.log_error(e, context = "Retrieval metrics evaluation")
            raise

    def _calculate_precision_at_k(self, retrieved: List[str], relevant: set) -> float:
        """
        Precision@K: Proportion of retieved documents that are relevant

        Formula: Relevant Retrieved / K
        """
        if not retrieved:
            return 0.0
        
        relevant_retrieved = sum(1 for doc_id in retrieved if doc_id in relevant)
        return relevant_retrieved / len(retrieved)
    
    def _calculate_recall_at_k(self, retrieved: List[str], relevant: set) -> float:
        """
        Recall@K: Proportion of relevant documents that are retrieved

        Formula: Relevant Retrieved / Total Relevant
        """
        if not relevant:
            return 0.0

        relevant_retrieved = sum(1 for doc_id in retrieved if doc_id in relevant)
        return relevant_retrieved / len(relevant)

    def _calculate_mrr(self, retrieved: List[str], relevant: set) -> float:
        """
        Mean Reciprocal Rank: Reciprocal of rank of first relevant document

        Formula: 1 / Ranl of First Relevant Document

        Example: First relevant doc at position 3 -> MRR = 1/3 = 0.333
        """
        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                return 1.0 / i
            
        return 0.0
    
    def _calculate_ndcg(self, results: List[Dict], relevant: set) -> float:
        """
        Normalized Discounted Cumulative Gain

        Consider both relevance and ranking position
        """
        if not results:
            return 0.0
        
        # DCG: Sum of (relevance / log2(rank + 1))
        dcg = 0.0
        for i, doc in enumerate(results, 1):
            relevance = 1.0 if doc['id'] in relevant else 0.0
            dcg += relevance / np.log2(i + 1)

        # IDCG: DCG of perfect ranking
        ideal_dcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(relevant) + 1, len(results) + 1)))

        if ideal_dcg == 0:
            return 0.0
        
        return dcg / ideal_dcg
    
    def _calculate_f1(self, precision: float, recall: float) -> float:
        """
        F1 Score: Harmonic search of Precision and Recall

        Formula: 2 * (P * R) / (P + R)
        """
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    # PART 2: RAGAS GENERATION METRICS

    def evaluate_generation_metrics(self, test_cases: List[Dict]) -> Dict:
        """
        Evaluate generation quality using RAGAS

        Args:
            test_cases: List of test cases
                [
                    {
                        "question": "What is diabetes?",
                        "ground_truth": "Diabetes is a chronic disease..."
                    },
                    ...
                ]
        Returns:
            Dictionary with RAGAS metrics
        """
        try:
            self.logger.logger.info("=" * 60)
            self.logger.logger.info("EVALUATING GENERATION METRICS (RAGAS)")
            self.logger.logger.info("=" * 60)

            # Prepare dataset
            self.logger.logger.info("\nPreparing dataset for RAGAS...")
            dataset_dict = self._prepare_ragas_dataset(test_cases)

            # Convert to RAGAS Dataset
            dataset = Dataset.from_dict(dataset_dict)

            self.logger.logger.info(f"Dataset prepared with {len(dataset)} samples")

            # Define RAGAS metrics
            metrics = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ]

            # Use Groq LLM and local HuggingFace embeddings (avoids OpenAI quota)
            ragas_llm = LangchainLLMWrapper(
                ChatGroq(model=Config.GROQ_MODEL, api_key=Config.GROQ_API_KEY)
            )
            ragas_embeddings = LangchainEmbeddingsWrapper(
                HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            )

            self.logger.logger.info(f"\nEvaluating RAGAS metrics...")
            self.logger.logger.info(f"  Metrics: {[m.name for m in metrics]}")
            self.logger.logger.info("   (This may take several minutes...)")

            # Run RAGAS evaluation
            results = evaluate(
                dataset = dataset,
                metrics = metrics,
                llm = ragas_llm,
                embeddings = ragas_embeddings,
            )

            # Convert to dictionary
            results_df = results.to_pandas()

            # Calculate summary
            summary = {
                'faithfulness': float(results_df['faithfulness'].mean()),
                'answer_relevancy': float(results_df['answer_relevancy'].mean()),
                'context_precision': float(results_df['context_precision'].mean()),
                'context_recall': float(results_df['context_recall'].mean()),
            }

            # Calculate overall generation score
            summary['overall_generation_score'] = float(np.mean(list(summary.values())))

            self.logger.logger.info("\nGeneration metrics evaluation completed")
            self.logger.logger.info(f"\nGeneration metrics Summary:")
            for metric, score in summary.items():
                if metric != 'overall_generation_score':
                    self.logger.logger.info(f"  {metric:25s}: {score:.4f}")
            self.logger.logger.info(f" {'Overall Generation score':25s}: {summary['overall_generation_score']:.4f}")

            return {
                'summary': summary,
                'detailed_results': results_df.to_dict()
            }
        
        except Exception as e:
            self.logger.log_error(e, context = "Generation metrics evaluation")
            raise

    def _prepare_ragas_dataset(self, test_cases: List[Dict]) -> Dict:
        """Prepare dataset in RAGAS format"""
        questions = []
        answers = []
        contexts = []
        ground_truths = []

        for i, test_case in enumerate(test_cases):
            question = test_case['question']
            ground_truth = test_case.get('ground_truth', '')

            # Query RAG system
            result = self.retriever.query(
                question = question,
                top_k = 5,
                include_sources = True
            )

            answer = result['answer']
            sources = result.get('sources', [])

            # Extract contexts
            context_list = []
            for source in sources:
                text = source.get('text_preview', source.get('text', ''))
                if text:
                    context_list.append(text)

            questions.append(question)
            answers.append(answer)
            contexts.append(context_list)
            ground_truths.append(ground_truth)

            if (i + 1) % 5 == 0:
                self.logger.logger.info(f"  Progress: {i + 1}/{len(test_cases)}")

        return {
            'question': questions,
            'answer': answers,
            'contexts': contexts,
            'ground_truth': ground_truths
        }
    
    # PART 3: COMPLETE END-TO-END EVALUATION

    def evaluate_complete_rag(self, test_cases: List[Dict], k_values: List[int] = [3, 5, 10]) -> Dict:
        """
        Complete end-to-end RAG evaluation

        Combines:
        1. Retrieval metrics (Precision, Recall, MRR, F1, NDCG)
        2. Generation metrics (RAGAS: Faithfulness, Relevancy etc.)

        Args:
            test_cases: Comprehensive test cases with both retrieval and generation ground truth
            k_values: K values for retrieval metrics

        Returns:
            Complete evaluation results
        """
        try:
            self.logger.logger.info("=" * 60)
            self.logger.logger.info("COMPLETE RAG EVALUATION")
            self.logger.logger.info("=" * 60)
            self.logger.logger.info(f"Test Cases: {len(test_cases)}")
            self.logger.logger.info(f"Timestamp: {datetime.now().isoformat()}")

            # Part 1: Retrieval Metrics
            self.logger.logger.info("\n[1/2] Evaluating Retrieval Metrics...")
            retrieval_results = self.evaluate_retrieval_metrics(test_cases = test_cases, k_values = k_values)

            # Part 2: Generation Metrics
            self.logger.logger.info("\n[2/2] Evaluating Generation Metrics...")
            generation_results = self.evaluate_generation_metrics(test_cases=test_cases)

            complete_results = {
                'timestamp': datetime.now().isoformat(),
                'num_test_cases': len(test_cases),
                'retrieval_metrics': retrieval_results,
                'generation_metrics': generation_results,
            }

            # Calculate overall RAG score
            # Weighted average: 40% retrieval, 60% generation
            retrieval_score = retrieval_results['k_5']['f1_score'] # Use K=5 for overall
            generation_score = generation_results['summary']['overall_generation_score']

            overall_rag_score = (retrieval_score * 0.4) + (generation_score * 0.6)
            complete_results['overall_rag_score'] = float(overall_rag_score)

            self._print_complete_summary(complete_results)

            return complete_results
        
        except Exception as e:
            self.logger.log_error(e, context = "Complete RAG evaluation")
            raise

    def _print_complete_summary(self, results: Dict):
        """Print comprehensive evaluation summary"""
        print("\n" + "="*60)
        print("COMPLETE RAG EVALUATION RESULTS")
        print("=" * 60)

        print(f"\nTest Cases: {results['num_test_cases']}")
        print(f"Timestamp: {results['timestamp']}")

        print("\n" + "-" * 60)
        print("RETRIEVAL METRICS")
        print("-" * 60)

        for k_key, metrics in results['retrieval_metrics'].items():
            k = k_key.split('_')[1]
            print(f"\nK = {k}:")
            print(f"    Precision@{k:2s} : {metrics['precision@k']: .4f}")
            print(f"    Recall@{k:2s} : {metrics['recall@k']: .4f}")
            print(f"    MRR : {metrics['mrr']: .4f}")
            print(f"    NDCG@{k:2s} : {metrics['ndcg@k']: .4f}")
            print(f"    F1 Score : {metrics['f1_score']: .4f}")

        print("\n" + "-" * 60)
        print("GENERATION METRICS (RAGAS)")
        print("-" * 60)

        gen_summary = results['generation_metrics']['summary']

        print(f"\n  Faithfulness           : {gen_summary['faithfulness']:.4f}")
        print(f"  Answer Relevancy       : {gen_summary['answer_relevancy']:.4f}")
        print(f"  Context Precision      : {gen_summary['context_precision']:.4f}")
        print(f"  Context Recall         : {gen_summary['context_recall']:.4f}")
        
        print("\n" + "="*60)
        print(f"OVERALL RAG SCORE: {results['overall_rag_score']:.4f}")
        print("="*60)
        
        # Interpretation
        score = results['overall_rag_score']
        if score >= 0.9:
            rating = "✓ EXCELLENT - Production Ready"
        elif score >= 0.7:
            rating = "○ GOOD - Minor improvements possible"
        elif score >= 0.5:
            rating = "△ FAIR - Needs optimization"
        else:
            rating = "✗ POOR - Significant issues"
        
        print(f"\nRating: {rating}\n")

    # PART 4: SAVE AND EXPORT

    def save_results(self, results: Dict, output_dir: str = "evaluation/results"):
        """Save evaluation results to JSON and CSV"""
        try:
            os.makedirs(output_dir, exist_ok = True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save JSON
            json_file = f"{output_dir}/complete_evaluation_{timestamp}.json"
            with open(json_file, 'w') as f:
                json.dump(results, f, indent = 2)

            self.logger.logger.info(f" JSON saved: {json_file}")

            # Save summary CSV
            summary_data = {
                'Metric': [],
                'Score': []
            }
            
            # Retrieval metrics (K=5)
            retrieval_k5 = results['retrieval_metrics']['k_5']
            summary_data['Metric'].extend([
                'Precision@5',
                'Recall@5',
                'MRR',
                'NDCG@5',
                'F1 Score (Retrieval)'
            ])
            summary_data['Score'].extend([
                retrieval_k5['precision@k'],
                retrieval_k5['recall@k'],
                retrieval_k5['mrr'],
                retrieval_k5['ndcg@k'],
                retrieval_k5['f1_score']
            ])
            
            # Generation metrics
            gen_summary = results['generation_metrics']['summary']
            summary_data['Metric'].extend([
                'Faithfulness',
                'Answer Relevancy',
                'Context Precision',
                'Context Recall',
            ])
            summary_data['Score'].extend([
                gen_summary['faithfulness'],
                gen_summary['answer_relevancy'],
                gen_summary['context_precision'],
                gen_summary['context_recall'],
            ])
            
            # Overall
            summary_data['Metric'].append('Overall RAG Score')
            summary_data['Score'].append(results['overall_rag_score'])
            
            # Save CSV
            df = pd.DataFrame(summary_data)
            csv_file = f"{output_dir}/summary_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            
            self.logger.logger.info(f"✓ CSV saved: {csv_file}")
            
        except Exception as e:
            self.logger.log_error(e, context="Saving evaluation results")
            raise

class MedicalTestDataset:
    """Generate comprehensive test dataset with ground truth"""

    def __init__(self):
        self.logger = MedicalRAGLogger(__name__)

    def generate_comprehensive_test_cases(self) -> List[Dict]:
        """
        Generate test cases with both retrieval and generation ground truth

        Returns:
            List of comprehensive test cases
        """            
        test_cases = [
            # Test Case 1: Diabetes - Definition
            {
                "question": "What is diabetes?",
                "relevant_doc_ids": [],  # Add actual doc IDs from your vector DB
                "ground_truth": "Diabetes is a chronic metabolic disease characterized by elevated blood glucose levels. It occurs when the pancreas doesn't produce enough insulin or when the body cannot effectively use the insulin it produces.",
                "category": "endocrine",
                "expected_keywords": ["chronic", "blood glucose", "insulin", "pancreas"]
            },
            
            # Test Case 2: Diabetes - Symptoms
            {
                "question": "What are the main symptoms of diabetes?",
                "relevant_doc_ids": [],
                "ground_truth": "The main symptoms of diabetes include increased thirst (polydipsia), frequent urination (polyuria), extreme fatigue, blurred vision, slow-healing wounds, and unexplained weight loss.",
                "category": "endocrine",
                "expected_keywords": ["thirst", "urination", "fatigue", "vision", "wounds"]
            },
            
            # Test Case 3: Diabetes - Treatment
            {
                "question": "How is diabetes treated?",
                "relevant_doc_ids": [],
                "ground_truth": "Diabetes treatment includes insulin therapy for Type 1, oral medications for Type 2, dietary modifications to control blood sugar, regular exercise, blood glucose monitoring, and lifestyle changes.",
                "category": "endocrine",
                "expected_keywords": ["insulin", "medication", "diet", "exercise", "monitoring"]
            },
            
            # Test Case 4: Heart Attack - Causes
            {
                "question": "What causes a heart attack?",
                "relevant_doc_ids": [],
                "ground_truth": "A heart attack occurs when blood flow to part of the heart muscle is blocked, usually by a blood clot in a coronary artery. This blockage prevents oxygen-rich blood from reaching the heart tissue.",
                "category": "cardiovascular",
                "expected_keywords": ["blood flow", "blockage", "clot", "coronary artery"]
            },
            
            # Test Case 5: Heart Attack - Symptoms
            {
                "question": "What are the warning signs of a heart attack?",
                "relevant_doc_ids": [],
                "ground_truth": "Heart attack warning signs include chest pain or discomfort, pain in the arms, back, neck, jaw or stomach, shortness of breath, cold sweat, nausea, and lightheadedness.",
                "category": "cardiovascular",
                "expected_keywords": ["chest pain", "shortness of breath", "sweating", "nausea"]
            },
            
            # Test Case 6: Hypertension - Treatment
            {
                "question": "How is high blood pressure treated?",
                "relevant_doc_ids": [],
                "ground_truth": "Hypertension treatment includes lifestyle modifications such as reducing sodium intake, regular exercise, maintaining healthy weight, limiting alcohol, and medications like ACE inhibitors, beta-blockers, or diuretics when necessary.",
                "category": "cardiovascular",
                "expected_keywords": ["lifestyle", "sodium", "exercise", "medication", "ACE inhibitors"]
            },
            
            # Test Case 7: Pneumonia
            {
                "question": "What is pneumonia?",
                "relevant_doc_ids": [],
                "ground_truth": "Pneumonia is an infection that causes inflammation of the air sacs in one or both lungs. The air sacs may fill with fluid or pus, causing symptoms like cough with phlegm, fever, chills, and difficulty breathing.",
                "category": "respiratory",
                "expected_keywords": ["infection", "lungs", "inflammation", "cough", "fever"]
            },
            
            # Test Case 8: Alzheimer's
            {
                "question": "What is Alzheimer's disease?",
                "relevant_doc_ids": [],
                "ground_truth": "Alzheimer's disease is a progressive neurological disorder that causes brain cells to die, leading to memory loss, cognitive decline, and behavioral changes. It is the most common cause of dementia in older adults.",
                "category": "neurological",
                "expected_keywords": ["brain", "memory loss", "cognitive", "dementia", "progressive"]
            },
            
            # Test Case 9: Complex Query
            {
                "question": "What is the relationship between diabetes and heart disease?",
                "relevant_doc_ids": [],
                "ground_truth": "Diabetes significantly increases the risk of cardiovascular disease. High blood sugar levels damage blood vessels and nerves controlling the heart, leading to atherosclerosis, coronary artery disease, heart attacks, and strokes.",
                "category": "endocrine",
                "expected_keywords": ["cardiovascular", "blood vessels", "atherosclerosis", "risk"]
            },
            
            # Test Case 10: Prevention
            {
                "question": "How can Type 2 diabetes be prevented?",
                "relevant_doc_ids": [],
                "ground_truth": "Type 2 diabetes prevention includes maintaining healthy weight through balanced diet and regular exercise, avoiding excessive sugar and refined carbohydrates, not smoking, limiting alcohol, and regular health screenings.",
                "category": "endocrine",
                "expected_keywords": ["prevention", "diet", "exercise", "weight", "screening"]
            }
        ]
            
        self.logger.logger.info(f"Generated {len(test_cases)} comprehensive test cases")

        return test_cases
    

# Usage/Testing
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("COMPREHENSIVE RAG EVALUATOR")
    print("Retrieval Metrics + RAGAS Generation Metrics")
    print("=" * 60)

    # Step 1: Generate test dataset
    print("Step 1: Generating test dataset...")
    print("-" * 60)

    dataset_gen = MedicalTestDataset()
    test_cases = dataset_gen.generate_comprehensive_test_cases()

    print(f"Generated {len(test_cases)} test cases\n")

    # Step 2: Initialize evaluator
    print("Step 2: Initializing comprehensive evaluator...")
    print("-" * 60)

    try:
        evaluator = ComprehensiveRAGEvaluator()
        print("Evaluator initialized\n")
    except Exception as e:
        print(f"x Error: {e}")
        print("\nMake sure:")
        print("  1. pip install ragas datasets pandas numpy")
        print("  2. RAG system is running")
        print("  3. All API keys configured")
        exit(1)

    # Step 3: Run complete evaluation
    print("Step 3: Running complete evaluation...")
    print("-" * 60)
    print("(This will take several minutes...)\n")

    try:
        results = evaluator.evaluate_complete_rag(
            test_cases = test_cases,
            k_values = [3, 5, 10]
        )

        # Save results
        evaluator.save_results(results)

        print("\nEvalaution completed successfully!")

    except Exception as e:
        print(f"\n Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("EVALUATION COMPLETED")
    print("="*60)
    
    print("\nResults saved to:")
    print("  - evaluation/results/complete_evaluation_*.json")
    print("  - evaluation/results/summary_*.csv")
    
    print("\nNext steps:")
    print("  1. Review metrics in CSV file")
    print("  2. Identify weak areas")
    print("  3. Optimize RAG configuration")
    print("  4. Re-evaluate improvements")