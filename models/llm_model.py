from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from rich import print as pprint

import os
from dotenv import load_dotenv
# import sys
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config'))

from langchain.agents import (AgentExecutor, create_openai_functions_agent)
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models'))

load_dotenv()

class SummaryModel:
    """
    SummaryModel - 負責摘要生成，使用 OpenAI GPT 和 LangChain 進行專業新聞摘要。
    """
    def __init__(self):
        """
        初始化摘要模型，並加載可用工具。
        """
        self.chat_model = ChatOpenAI(model=os.getenv('OPENAI_MODEL_NAME', 'gpt-4o-mini'), api_key=os.getenv('OPENAI_API_KEY'))

    def get_final_summary_prompt(self):
        """
        建立專業摘要的 Prompt。
        :input_text: 原始新聞文本
        :return: ChatPromptTemplate 物件
        """
        prompt_content = [
            ('system', '你是一位專業的新聞編輯，負責為不同領域的新聞生成高品質摘要'),
            ('system', '請針對以下新聞文本生成「新聞摘要」，並確保包含：'),
            ('system', '1. 時間：事件發生或報導的時間'),
            ('system', '2. 地點：事件發生的地點或相關地區'),
            ('system', '3. 人物：事件涉及的核心人物、機構或企業'),
            ('system', '4. 事件：事件的核心內容、技術、背景或原因'),
            ('system', '5. 影響：事件對產業、社會、經濟或其他方面的影響'),
            ('system', '6. 數據：若有數據，請保留關鍵數字與指標'),
            ('system', '請將摘要限制在 1,000 字內，並保持專業簡潔的語氣'),
            ('system', '摘要範例格式：'),
            ('system', '報導時間：(時間)\n發生地點：(地點)\n牽涉人物：(人物)\n事件說明：(事件)\n事件影響：(影響)\n相關數據：(數據)'),
            ('system', '**重要**：摘要內容必須格依照提供的新聞文本撰寫，嚴禁生成文本中未提及的資訊或虛構內容'),
            ('human', '以下為新聞文本，請直接生成摘要：{input_text}')
        ]
        return ChatPromptTemplate.from_messages(prompt_content)

    def generate_final_summary(self, news_text):
        """
        產生新聞摘要。
        :param text: 原始新聞文本
        :return: 生成的摘要結果
        """       
        if isinstance(news_text, list):
            news_text = self.combine_paragraph_summary(news_text)
        prompt = self.get_final_summary_prompt()
        try:
            response = self.chat_model.invoke(prompt.format(input_text=news_text))
            return response.content

        except Exception as e:
            pprint(f"[red]Error during summary generation: {e}[/red]")
            return "抱歉，目前無法生成摘要，請稍後再試。"
    
    def get_paragraph_summary_prompt(self):
        """
        建立常文本段落摘要的 Prompt。
        :param_text: 原始新聞文本
        :return: ChatPromptTemplate 物件
        """
        prompt_content = [
            ('system', '你是一位專業的新聞編輯，負責為長篇新聞文本生成高品質摘要。'),
            ('system', '以下為新聞文本的第 {segment_number} 段，請生成本段的重點摘要，避免重複或缺少重要資訊。'),
            ('system', '若段落中包含時間(事件發生或報導的時間)、地點(事件發生的地點或相關地區)、人物(事件涉及的核心人物、機構或企業)、事件(事件的核心內容、技術、背景或原因)、影響(事件對產業、社會、經濟或其他方面的影響)或數據(若有數據，請保留關鍵數字與指標)都必須記錄下來。'),
            ('system', '請確保摘要內容保持邏輯連貫，並與其他段落的主題一致。'),
            ('system', '請將摘要限制在 200 字內，並保持專業簡潔的語氣'),
            ('system', '**重要**：僅依照提供的文本撰寫摘要，不產生虛構內容。'),
            ('human', '新聞文本段落：{content}')
        ]
        return ChatPromptTemplate.from_messages(prompt_content)

    def generate_paragraph_summary(self, segment_number, content):
        """
        產生新聞摘要。
        :param text: 原始新聞文本
        :return: 生成的摘要結果
        """
        prompt = self.get_paragraph_summary_prompt()
        try:
            response = self.chat_model.invoke(prompt.format(segment_number=segment_number, content=content))
            return response.content

        except Exception as e:
            pprint(f"[red]Error during summary generation: {e}[/red]")
            return "抱歉，目前無法生成摘要，請稍後再試。"

    def combine_paragraph_summary(self, paragraph_list):
        summaries = []
        for segment_number, content in enumerate(paragraph_list):
            summary = self.generate_paragraph_summary(segment_number+1, content)
            summaries.append(summary)
        return '\n'.join(summaries)
            
if __name__ == "__main__":
    model = SummaryModel()
    news_text = "高安縣鄉鎮企業重視人才和智力資源開發..."
    news_text = [news_text, news_text]
    summary = model.generate_final_summary(news_text)
    pprint(summary)
    print(len(summary))
