# Run ga.py
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from BASE import ga

def main():
    print("메인 스크립트 시작.")
    ga.main("benchmark/plateau1.py") # run target file on ga.py
    print("Main script finished.")

if __name__ == "__main__":
    main()