import os
import csv

def get_kept_resume_points(csv_path=None):
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), '../resume/revised_resume_points_similarity.csv')
    results = []
    with open(csv_path, 'r') as f:
        reader = list(csv.DictReader(f))
        for i in range(0, len(reader), 2):
            row1 = reader[i]
            row2 = reader[i+1] if i+1 < len(reader) else None
            if not row2:
                # Only keep if 'original' is NOT in the kept point
                if 'original' not in row1['Revised Resume Point'].lower():
                    results.append({
                        'kept': row1['Revised Resume Point'],
                        'discarded': None,
                        'kept_score': float(row1.get('Similarity Score', 0)),
                        'discarded_score': None
                    })
                continue
            score1 = float(row1.get('Similarity Score', 0))
            score2 = float(row2.get('Similarity Score', 0))
            point1 = row1.get('Revised Resume Point', '')
            point2 = row2.get('Revised Resume Point', '')
            # Determine which point is kept
            if score1 > score2:
                kept, discarded, kept_score, discarded_score = point1, point2, score1, score2
            elif score2 > score1:
                kept, discarded, kept_score, discarded_score = point2, point1, score2, score1
            else:
                # If scores are equal, keep the one with 'original' in the text
                if 'original' in point1.lower():
                    kept, discarded, kept_score, discarded_score = point1, point2, score1, score2
                else:
                    kept, discarded, kept_score, discarded_score = point2, point1, score2, score1
            # Only keep if 'original' is NOT in the kept point
            if 'original' not in kept.lower():
                results.append({'kept': kept, 'discarded': discarded, 'kept_score': kept_score, 'discarded_score': discarded_score})
    return results

if __name__ == "__main__":
    for result in get_kept_resume_points():
        print(f"KEPT: {result['kept']} (Score: {result['kept_score']})\nDISCARDED: {result['discarded']} (Score: {result['discarded_score']})\n---")
