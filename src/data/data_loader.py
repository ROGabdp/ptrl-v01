"""
DataLoader - S&P 500 股票資料下載與管理模組

功能:
- 從 Wikipedia 取得 S&P 500 成分股清單
- 使用 yfinance 下載股票 OHLCV 資料
- CSV 快取機制，避免重複下載
- 增量更新，只下載新資料
"""

import os
import json
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from loguru import logger


class DataLoader:
    """
    S&P 500 股票資料下載與管理器
    
    使用方式:
        loader = DataLoader(config)
        loader.download_all(start_date='2005-02-25', end_date='2023-10-15')
        df = loader.load_symbol('AAPL')
    """
    
    def __init__(self, config: dict):
        """
        初始化 DataLoader
        
        Args:
            config: 設定字典，需包含:
                - data_dir: 原始資料目錄 (預設 'data/raw/')
                - processed_dir: 處理後資料目錄 (預設 'data/processed/')
                - index_symbol: 指數代碼 (預設 '^GSPC')
                - symbols_file: 股票清單檔案路徑
        """
        self.data_dir = config.get('data_dir', 'data/raw/')
        self.processed_dir = config.get('processed_dir', 'data/processed/')
        self.index_symbol = config.get('index_symbol', '^GSPC')
        self.symbols_file = config.get('symbols_file', 'config/sp500_symbols.txt')
        self.cache_status_file = os.path.join(self.data_dir, 'cache_status.json')
        
        # 確保目錄存在
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.symbols_file), exist_ok=True)
        
        logger.info(f"DataLoader 初始化完成 - 資料目錄: {self.data_dir}")
    
    # =========================================================================
    # S&P 500 成分股管理
    # =========================================================================
    
    def get_sp500_symbols(self, save: bool = True) -> List[str]:
        """
        從 Wikipedia 取得最新 S&P 500 成分股清單
        
        Args:
            save: 是否儲存至本地檔案
            
        Returns:
            股票代碼清單
        """
        import requests
        from io import StringIO
        
        logger.info("正在從 Wikipedia 取得 S&P 500 成分股清單...")
        
        try:
            # 使用 requests 並設定 User-Agent 來避免 403 錯誤
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            # 從 HTML 內容解析表格
            tables = pd.read_html(StringIO(response.text))
            df = tables[0]
            
            # 取得股票代碼 (處理特殊字元，如 BRK.B -> BRK-B)
            symbols = df['Symbol'].str.replace('.', '-', regex=False).tolist()
            
            logger.info(f"成功取得 {len(symbols)} 支股票代碼")
            
            # 儲存至本地檔案
            if save:
                with open(self.symbols_file, 'w') as f:
                    f.write('\n'.join(symbols))
                logger.info(f"股票清單已儲存至: {self.symbols_file}")
            
            return symbols
            
        except Exception as e:
            logger.error(f"取得 S&P 500 成分股失敗: {e}")
            raise
    
    def load_symbols_list(self) -> List[str]:
        """
        從本地檔案載入股票清單
        
        Returns:
            股票代碼清單
        """
        if not os.path.exists(self.symbols_file):
            logger.warning(f"股票清單不存在，正在從網路下載...")
            return self.get_sp500_symbols()
        
        with open(self.symbols_file, 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
        
        logger.info(f"已載入 {len(symbols)} 支股票")
        return symbols
    
    # =========================================================================
    # 資料下載
    # =========================================================================
    
    def download_symbol(self, symbol: str, start_date: str, end_date: str,
                        save: bool = True) -> Optional[pd.DataFrame]:
        """
        下載單一股票資料並存為 CSV
        
        Args:
            symbol: 股票代碼
            start_date: 開始日期 (YYYY-MM-DD)
            end_date: 結束日期 (YYYY-MM-DD)
            save: 是否儲存至 CSV
            
        Returns:
            OHLCV DataFrame，下載失敗返回 None
        """
        try:
            # 下載資料
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
            
            if df.empty:
                logger.warning(f"{symbol}: 無資料")
                return None
            
            # 重置索引，將日期變成欄位
            df = df.reset_index()
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            
            # 只保留需要的欄位
            columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            df = df[columns]
            
            # 儲存至 CSV
            if save:
                csv_path = os.path.join(self.data_dir, f'{symbol}.csv')
                df.to_csv(csv_path, index=False)
            
            return df
            
        except Exception as e:
            logger.error(f"{symbol}: 下載失敗 - {e}")
            return None
    
    def download_index(self, start_date: str, end_date: str,
                       save: bool = True) -> Optional[pd.DataFrame]:
        """
        下載指數資料 (S&P 500)
        
        Args:
            start_date: 開始日期
            end_date: 結束日期
            save: 是否儲存至 CSV
            
        Returns:
            指數 OHLCV DataFrame
        """
        logger.info(f"正在下載指數 {self.index_symbol}...")
        df = self.download_symbol(self.index_symbol, start_date, end_date, save=False)
        
        if df is not None and save:
            # 將 ^GSPC 儲存為 GSPC.csv (移除特殊字元)
            filename = self.index_symbol.replace('^', '') + '.csv'
            csv_path = os.path.join(self.data_dir, filename)
            df.to_csv(csv_path, index=False)
            logger.info(f"指數資料已儲存至: {csv_path}")
        
        return df
    
    def download_all(self, start_date: str, end_date: str,
                     symbols: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        下載所有 S&P 500 成分股資料
        
        Args:
            start_date: 開始日期
            end_date: 結束日期
            symbols: 自訂股票清單 (預設使用 S&P 500)
            
        Returns:
            {symbol: DataFrame} 字典
        """
        if symbols is None:
            symbols = self.load_symbols_list()
        
        logger.info(f"開始下載 {len(symbols)} 支股票，期間: {start_date} ~ {end_date}")
        
        results = {}
        failed = []
        
        # 下載指數
        self.download_index(start_date, end_date)
        
        # 下載個股
        for symbol in tqdm(symbols, desc="下載進度"):
            df = self.download_symbol(symbol, start_date, end_date)
            if df is not None:
                results[symbol] = df
            else:
                failed.append(symbol)
        
        # 更新快取狀態
        self._update_cache_status(start_date, end_date)
        
        logger.info(f"下載完成: 成功 {len(results)} 支，失敗 {len(failed)} 支")
        if failed:
            logger.warning(f"下載失敗的股票: {failed[:10]}{'...' if len(failed) > 10 else ''}")
        
        return results
    
    # =========================================================================
    # 資料更新 (增量下載)
    # =========================================================================
    
    def update_symbol(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        更新單一股票資料 (增量下載)
        
        Args:
            symbol: 股票代碼
            
        Returns:
            更新後的完整 DataFrame
        """
        csv_path = os.path.join(self.data_dir, f'{symbol}.csv')
        
        # 如果不存在，完整下載
        if not os.path.exists(csv_path):
            logger.info(f"{symbol}: 無現有資料，執行完整下載")
            return self.download_symbol(symbol, '2005-01-01', datetime.now().strftime('%Y-%m-%d'))
        
        # 讀取現有資料
        existing_df = pd.read_csv(csv_path, parse_dates=['Date'])
        last_date = existing_df['Date'].max()
        
        # 計算需要下載的日期範圍
        start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        # 如果已是最新，無需更新
        if last_date.date() >= datetime.now().date() - timedelta(days=1):
            logger.debug(f"{symbol}: 資料已是最新")
            return existing_df
        
        # 下載新資料
        logger.info(f"{symbol}: 更新資料 {start_date} ~ {end_date}")
        new_df = self.download_symbol(symbol, start_date, end_date, save=False)
        
        if new_df is not None and not new_df.empty:
            # 合併資料
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['Date']).sort_values('Date')
            
            # 儲存
            combined_df.to_csv(csv_path, index=False)
            return combined_df
        
        return existing_df
    
    def update_all(self) -> Dict[str, pd.DataFrame]:
        """
        更新所有已下載的股票至最新日期
        
        Returns:
            {symbol: DataFrame} 字典
        """
        symbols = self.load_symbols_list()
        logger.info(f"開始更新 {len(symbols)} 支股票...")
        
        results = {}
        
        # 更新指數
        index_filename = self.index_symbol.replace('^', '') + '.csv'
        if os.path.exists(os.path.join(self.data_dir, index_filename)):
            self.update_symbol(self.index_symbol.replace('^', ''))
        
        # 更新個股
        for symbol in tqdm(symbols, desc="更新進度"):
            df = self.update_symbol(symbol)
            if df is not None:
                results[symbol] = df
        
        # 更新快取狀態
        self._update_cache_status()
        
        logger.info(f"更新完成: {len(results)} 支股票")
        return results
    
    # =========================================================================
    # 資料載入
    # =========================================================================
    
    def load_symbol(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        從 CSV 載入單一股票資料
        
        Args:
            symbol: 股票代碼
            
        Returns:
            OHLCV DataFrame (Date 為索引)
        """
        csv_path = os.path.join(self.data_dir, f'{symbol}.csv')
        
        if not os.path.exists(csv_path):
            logger.warning(f"{symbol}: CSV 不存在")
            return None
        
        df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
        return df
    
    def load_index(self) -> Optional[pd.DataFrame]:
        """
        載入指數資料
        
        Returns:
            指數 OHLCV DataFrame
        """
        filename = self.index_symbol.replace('^', '') + '.csv'
        csv_path = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(csv_path):
            logger.warning(f"指數 CSV 不存在: {csv_path}")
            return None
        
        df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
        return df
    
    def load_all(self, symbols: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        載入所有已下載的股票資料
        
        Args:
            symbols: 指定股票清單 (預設載入全部)
            
        Returns:
            {symbol: DataFrame} 字典
        """
        if symbols is None:
            symbols = self.load_symbols_list()
        
        results = {}
        
        for symbol in tqdm(symbols, desc="載入資料"):
            df = self.load_symbol(symbol)
            if df is not None:
                results[symbol] = df
        
        logger.info(f"已載入 {len(results)} 支股票資料")
        return results
    
    # =========================================================================
    # 快取管理
    # =========================================================================
    
    def get_cache_status(self) -> pd.DataFrame:
        """
        取得快取狀態報告
        
        Returns:
            DataFrame with columns [symbol, start_date, end_date, rows, file_size_kb]
        """
        status_list = []
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.csv'):
                symbol = filename.replace('.csv', '')
                csv_path = os.path.join(self.data_dir, filename)
                
                try:
                    df = pd.read_csv(csv_path, parse_dates=['Date'])
                    file_size = os.path.getsize(csv_path) / 1024  # KB
                    
                    status_list.append({
                        'symbol': symbol,
                        'start_date': df['Date'].min().strftime('%Y-%m-%d'),
                        'end_date': df['Date'].max().strftime('%Y-%m-%d'),
                        'rows': len(df),
                        'file_size_kb': round(file_size, 2)
                    })
                except Exception as e:
                    logger.warning(f"無法讀取 {filename}: {e}")
        
        return pd.DataFrame(status_list)
    
    def clear_cache(self, symbol: str = None):
        """
        清除快取 (指定股票或全部)
        
        Args:
            symbol: 指定股票代碼，None 則清除全部
        """
        if symbol:
            csv_path = os.path.join(self.data_dir, f'{symbol}.csv')
            if os.path.exists(csv_path):
                os.remove(csv_path)
                logger.info(f"已刪除: {csv_path}")
        else:
            for filename in os.listdir(self.data_dir):
                if filename.endswith('.csv'):
                    os.remove(os.path.join(self.data_dir, filename))
            logger.info(f"已清除所有快取")
    
    def _update_cache_status(self, start_date: str = None, end_date: str = None):
        """更新快取狀態檔案"""
        status = {
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'start_date': start_date,
            'end_date': end_date,
            'total_files': len([f for f in os.listdir(self.data_dir) if f.endswith('.csv')])
        }
        
        with open(self.cache_status_file, 'w') as f:
            json.dump(status, f, indent=2)


# =============================================================================
# 使用範例
# =============================================================================

if __name__ == '__main__':
    # 設定 loguru
    logger.add("logs/data_loader.log", rotation="10 MB")
    
    # 設定
    config = {
        'data_dir': 'data/raw/',
        'processed_dir': 'data/processed/',
        'index_symbol': '^GSPC',
        'symbols_file': 'config/sp500_symbols.txt'
    }
    
    # 初始化
    loader = DataLoader(config)
    
    # 取得 S&P 500 成分股
    symbols = loader.get_sp500_symbols()
    print(f"取得 {len(symbols)} 支股票")
    
    # 下載前 5 支股票測試
    test_symbols = symbols[:5]
    loader.download_all(
        start_date='2020-01-01',
        end_date='2023-12-31',
        symbols=test_symbols
    )
    
    # 查看快取狀態
    status = loader.get_cache_status()
    print(status)
