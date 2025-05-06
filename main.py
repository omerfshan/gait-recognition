# train.py'den eğitim fonksiyonunu içe aktar
from train import run_training

def main():
    # Eğitimi çalıştır ve sonuçları al
    results = run_training()
    
    if results:
        # Sonuçları göster
        print(f"\nModel Doğruluk Oranı: {results['accuracy']:.2%}")
        
        # Her kişinin durumunu göster
        print("\nKişilerin Durumları:")
        for identifier, status in results['status_results'].items():
            print(f"{identifier}: {status}")

# Program başlangıç noktası
if __name__ == "__main__":
    main() 