"""
Deterministic query templates and scorer for LLM outage fallback.

When live model capacity across all LLM providers is exhausted (429s, network
outages, circuit-breaker trips), this module provides a deterministic floor:
matching common founder questions to pre-tested DuckDB SQL queries without any
external API calls or embeddings.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

# --------------------------------------------------------------------------- #
# Synonym dictionary and phrase mappings
# --------------------------------------------------------------------------- #

SYNONYMS: dict[str, str] = {
    # Receivables / Accounts Receivable / AR
    "receivable": "receivables",
    "receivables": "receivables",
    "outstanding": "outstanding",
    "ar": "receivables",
    "unpaid": "receivables",
    "due": "receivables",
    "overdue": "receivables",
    "owing": "outstanding",
    "owe": "outstanding",
    "exposure": "outstanding",
    # Won / Win / Deals won
    "win": "win",
    "won": "win",
    "wins": "win",
    "winning": "win",
    # Pipeline / Opportunities / Open deals
    "pipeline": "pipeline",
    "opportunities": "pipeline",
    "opportunity": "pipeline",
    "lead": "deals",
    "leads": "deals",
    "opps": "pipeline",
    "open": "pipeline",
    "book": "pipeline",
    # Billed / Invoiced / Invoices
    "billed": "billed",
    "billing": "billed",
    "invoiced": "billed",
    "invoicing": "billed",
    "invoice": "billed",
    "invoices": "billed",
    # Sector / Vertical / Industry
    "sector": "sector",
    "sectors": "sector",
    "vertical": "sector",
    "verticals": "sector",
    "industry": "sector",
    "industries": "sector",
    "segment": "sector",
    "segments": "sector",
    # Customer / Client / Account
    "customer": "customer",
    "customers": "customer",
    "client": "customer",
    "clients": "customer",
    "account": "customer",
    "accounts": "customer",
    # Owner / Rep
    "owner": "owner",
    "owners": "owner",
    "rep": "owner",
    "reps": "owner",
    "holds": "owner",
    "holding": "owner",
    # Work order / project language
    "job": "work_order",
    "jobs": "work_order",
    "project": "work_order",
    "projects": "work_order",
    # Completed / Finished
    "completed": "completed",
    "finished": "completed",
    "done": "completed",
    # Dead / Lost
    "dead": "dead",
    "lost": "dead",
    # Unbilled / Uninvoiced
    "unbilled": "unbilled",
    "uninvoiced": "unbilled",
    # Superlatives / Ranking
    "largest": "largest",
    "biggest": "largest",
    "highest": "largest",
    "top": "largest",
    "leader": "largest",
    "max": "largest",
    "single": "largest",
    # Averages
    "average": "average",
    "avg": "average",
    "mean": "average",
    "median": "average",
    # Fiscal Quarters
    "quarter": "quarter",
    "quarters": "quarter",
    "fq": "quarter",
    "fy": "quarter",
    # Age / Duration
    "oldest": "oldest",
    "longest": "oldest",
    "earliest": "oldest",
}

STOPWORDS: set[str] = {
    "what", "is", "our", "the", "and", "for", "of", "in", "to", "a", "an", "are", "we",
    "do", "have", "how", "much", "many", "does", "with", "by", "on", "all", "from",
    "which", "who", "show", "me", "give", "get", "list", "tell", "calculate", "summarise",
    "summarize", "find", "between", "each", "across", "per", "so", "far", "there",
    "been", "haven", "t", "s", "was", "were", "it", "its", "their", "them", "these",
    "those", "at", "into", "as", "out", "up", "down", "break", "carried", "carry",
    "money", "people", "still", "us", "using", "only", "worth",
}

# --------------------------------------------------------------------------- #
# Template Definition
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class QueryTemplate:
    id: str
    intent: str
    keywords: tuple[str, ...]
    sql: str
    required_keywords: tuple[str, ...] = ()
    disallowed_keywords: tuple[str, ...] = ()


TEMPLATES: list[QueryTemplate] = [
    # 1. Overall win rate
    QueryTemplate(
        id="overall_win_rate",
        intent="Overall closed-deal win rate",
        keywords=("win_rate", "win", "closed", "deals", "percentage", "rate", "overall", "decided", "share"),
        sql="""
            SELECT 
                ROUND(100.0 * SUM(CASE WHEN is_won THEN 1 ELSE 0 END) / COUNT(*), 2) AS win_rate_pct,
                SUM(CASE WHEN is_won THEN 1 ELSE 0 END) AS won_deals,
                SUM(CASE WHEN is_dead THEN 1 ELSE 0 END) AS dead_deals,
                COUNT(*) AS total_closed_deals
            FROM deals 
            WHERE is_won OR is_dead
        """,
        disallowed_keywords=("sector", "owner"),
    ),
    # 2. Win rate by sector
    QueryTemplate(
        id="win_rate_by_sector",
        intent="Win rate breakdown by sector",
        keywords=("win_rate", "win", "sector", "rate", "differ", "compare", "breakdown", "industry", "vertical", "overall"),
        sql="""
            SELECT 
                COALESCE(sector, 'Unknown') AS sector,
                SUM(CASE WHEN is_won THEN 1 ELSE 0 END) AS won_deals,
                SUM(CASE WHEN is_dead THEN 1 ELSE 0 END) AS dead_deals,
                COUNT(*) AS total_closed_deals,
                ROUND(100.0 * SUM(CASE WHEN is_won THEN 1 ELSE 0 END) / COUNT(*), 2) AS win_rate_pct
            FROM deals 
            WHERE is_won OR is_dead
            GROUP BY sector
            ORDER BY win_rate_pct DESC, total_closed_deals DESC
        """,
        required_keywords=("sector",),
    ),
    # 3. Open pipeline value (overall summary)
    QueryTemplate(
        id="open_pipeline_summary",
        intent="Total open pipeline volume and value",
        keywords=("pipeline", "total", "value", "raw", "live", "coverage", "unweighted", "carry", "summary", "aggregate"),
        sql="""
            SELECT 
                COUNT(*) AS open_deals_count,
                COUNT(deal_value) AS valued_deals_count,
                SUM(deal_value) AS raw_open_pipeline_value,
                SUM(weighted_value) AS weighted_pipeline_value
            FROM deals 
            WHERE deal_status = 'Open'
        """,
        disallowed_keywords=("sector", "owner", "quarter", "stage", "renewables", "mining", "railways", "powerline"),
    ),
    # 4. Pipeline by owner
    QueryTemplate(
        id="pipeline_by_owner",
        intent="Open pipeline grouped by deal owner",
        keywords=("pipeline", "owner", "manage", "rep", "distribution", "leader", "largest", "who", "highest", "ranking"),
        sql="""
            SELECT 
                owner_code,
                COUNT(*) AS open_deals_count,
                COUNT(deal_value) AS valued_deals_count,
                SUM(deal_value) AS raw_open_pipeline_value,
                SUM(weighted_value) AS weighted_pipeline_value
            FROM deals 
            WHERE deal_status = 'Open'
            GROUP BY owner_code
            ORDER BY raw_open_pipeline_value DESC NULLS LAST
        """,
        required_keywords=("owner",),
    ),
    # 5. Pipeline by sector
    QueryTemplate(
        id="pipeline_by_sector",
        intent="Open pipeline grouped by sector",
        keywords=("pipeline", "sector", "industry", "vertical", "distribution", "breakdown"),
        sql="""
            SELECT 
                COALESCE(sector, 'Unknown') AS sector,
                COUNT(*) AS open_deals_count,
                COUNT(deal_value) AS valued_deals_count,
                SUM(deal_value) AS raw_open_pipeline_value,
                SUM(weighted_value) AS weighted_pipeline_value
            FROM deals 
            WHERE deal_status = 'Open'
            GROUP BY sector
            ORDER BY raw_open_pipeline_value DESC NULLS LAST
        """,
        required_keywords=("sector",),
        disallowed_keywords=("receivables", "renewables", "mining", "railways", "powerline"),
    ),
    # 6. Pipeline by fiscal quarter
    QueryTemplate(
        id="pipeline_by_fiscal_quarter",
        intent="Open pipeline by tentative close fiscal quarter",
        keywords=("pipeline", "quarter", "close", "tentative", "scheduled", "timing", "fy"),
        sql="""
            SELECT 
                COALESCE(tentative_close_date_fq, 'Unknown') AS tentative_close_fiscal_quarter,
                COUNT(*) AS open_deals_count,
                COUNT(deal_value) AS valued_deals_count,
                SUM(deal_value) AS raw_open_pipeline_value
            FROM deals 
            WHERE deal_status = 'Open'
            GROUP BY tentative_close_date_fq
            ORDER BY tentative_close_fiscal_quarter
        """,
        required_keywords=("quarter",),
    ),
    # 7. Outstanding receivables total
    QueryTemplate(
        id="outstanding_receivables_total",
        intent="Total outstanding receivables and billing",
        keywords=("receivables", "outstanding", "total", "ar", "exposure", "unpaid", "balance", "amount"),
        sql="""
            SELECT 
                COUNT(*) AS total_work_orders,
                SUM(billed_incl_gst) AS total_billed_incl_gst,
                SUM(collected_incl_gst) AS total_collected_incl_gst,
                SUM(outstanding_incl_gst) AS total_outstanding_receivable
            FROM work_orders
        """,
        disallowed_keywords=("sector", "customer", "mining"),
    ),
    # 8. Receivables by sector
    QueryTemplate(
        id="receivables_by_sector",
        intent="Outstanding receivables grouped by sector",
        keywords=("receivables", "outstanding", "sector", "concentrated", "concentration", "breakdown", "industry", "vertical"),
        sql="""
            SELECT 
                COALESCE(sector, 'Unknown') AS sector,
                COUNT(*) AS work_orders_count,
                SUM(billed_incl_gst) AS total_billed_incl_gst,
                SUM(collected_incl_gst) AS total_collected_incl_gst,
                SUM(outstanding_incl_gst) AS outstanding_receivable
            FROM work_orders
            GROUP BY sector
            ORDER BY outstanding_receivable DESC NULLS LAST
        """,
        required_keywords=("sector",),
    ),
    # 9. Completed but uninvoiced work orders (row detail: includes item_id)
    QueryTemplate(
        id="completed_uninvoiced_work_orders",
        intent="Completed work orders with zero billed amount",
        keywords=("completed", "unbilled", "work_order", "uninvoiced", "jobs", "finished", "zero"),
        sql="""
            SELECT 
                item_id,
                deal_name,
                customer_code,
                sector,
                execution_status,
                amount_incl_gst,
                billed_incl_gst
            FROM work_orders
            WHERE execution_status = 'Completed' AND COALESCE(billed_incl_gst, 0) = 0
            ORDER BY amount_incl_gst DESC NULLS LAST
        """,
        required_keywords=("completed", "unbilled"),
    ),
    # 10. Billed vs collected summary
    QueryTemplate(
        id="billed_vs_collected_summary",
        intent="Aggregate contracted, billed, and collected revenue",
        keywords=("contracted", "billed", "collected", "total", "work_order", "revenue", "aggregate", "amounts", "billing"),
        sql="""
            SELECT 
                COUNT(*) AS total_work_orders,
                SUM(amount_incl_gst) AS total_contracted_incl_gst,
                SUM(billed_incl_gst) AS total_billed_incl_gst,
                SUM(collected_incl_gst) AS total_collected_incl_gst,
                SUM(unbilled_incl_gst) AS total_unbilled_incl_gst,
                SUM(outstanding_incl_gst) AS total_outstanding_incl_gst,
                ROUND(100.0 * SUM(collected_incl_gst) / NULLIF(SUM(billed_incl_gst), 0), 2) AS collection_pct
            FROM work_orders
        """,
        disallowed_keywords=("customer", "sector"),
    ),
    # 11. Top customers by contracted value
    QueryTemplate(
        id="top_customers_by_contracted_value",
        intent="Top customers ranked by total contracted work-order value",
        keywords=("customer", "largest", "contracted", "value", "work_order", "amount", "top", "highest"),
        sql="""
            SELECT 
                customer_code,
                COUNT(*) AS work_orders_count,
                SUM(amount_incl_gst) AS total_contracted_amount,
                SUM(billed_incl_gst) AS total_billed_amount,
                SUM(collected_incl_gst) AS total_collected_amount,
                SUM(outstanding_incl_gst) AS total_outstanding_balance
            FROM work_orders
            GROUP BY customer_code
            ORDER BY total_contracted_amount DESC NULLS LAST
            LIMIT 10
        """,
        required_keywords=("customer", "contracted"),
    ),
    # 12. Top customers by outstanding balance
    QueryTemplate(
        id="top_customers_by_outstanding_balance",
        intent="Top customers ranked by outstanding receivable balance",
        keywords=("customer", "largest", "receivables", "balance", "unpaid", "owed", "top", "highest"),
        sql="""
            SELECT 
                customer_code,
                COUNT(*) AS work_orders_count,
                SUM(outstanding_incl_gst) AS total_outstanding_balance,
                SUM(billed_incl_gst) AS total_billed_amount,
                SUM(collected_incl_gst) AS total_collected_amount
            FROM work_orders
            GROUP BY customer_code
            ORDER BY total_outstanding_balance DESC NULLS LAST
            LIMIT 10
        """,
        required_keywords=("customer", "receivables"),
    ),
    # 13. Deal counts by status
    QueryTemplate(
        id="deal_counts_by_status",
        intent="Deal distribution by status",
        keywords=("deal_status", "status", "count", "distribution", "mix", "deals", "breakdown"),
        sql="""
            SELECT 
                deal_status,
                COUNT(*) AS deal_count,
                ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM deals), 2) AS pct_of_total,
                COUNT(deal_value) AS valued_deals_count,
                SUM(deal_value) AS total_deal_value
            FROM deals
            GROUP BY deal_status
            ORDER BY deal_count DESC
        """,
    ),
    # 14. Single largest open deal (row detail: includes item_id)
    QueryTemplate(
        id="largest_open_deal",
        intent="Single largest open opportunity by deal value",
        keywords=("largest", "single", "pipeline", "deal", "value", "top", "leader", "biggest"),
        sql="""
            SELECT 
                item_id,
                deal_name,
                owner_code,
                client_code,
                sector,
                deal_value,
                deal_stage,
                tentative_close_date
            FROM deals
            WHERE deal_status = 'Open' AND deal_value IS NOT NULL
            ORDER BY deal_value DESC
            LIMIT 1
        """,
        required_keywords=("largest", "pipeline"),
    ),
    # 15. Top largest deals overall (row detail: includes item_id)
    QueryTemplate(
        id="top_deals_overall",
        intent="Top largest deals overall across all statuses",
        keywords=("largest", "deals", "overall", "value", "top", "high_value", "big"),
        sql="""
            SELECT 
                item_id,
                deal_name,
                owner_code,
                client_code,
                sector,
                deal_status,
                deal_stage,
                deal_value
            FROM deals
            WHERE deal_value IS NOT NULL
            ORDER BY deal_value DESC
            LIMIT 10
        """,
        required_keywords=("largest",),
        disallowed_keywords=("pipeline", "customer", "work_order"),
    ),
    # 16. Average deal size overall
    QueryTemplate(
        id="average_deal_size_overall",
        intent="Average deal size across all opportunities",
        keywords=("average", "deal", "size", "value", "mean", "deals"),
        sql="""
            SELECT 
                COUNT(deal_value) AS valued_deals_count,
                AVG(deal_value) AS avg_deal_value,
                MEDIAN(deal_value) AS median_deal_value,
                MIN(deal_value) AS min_deal_value,
                MAX(deal_value) AS max_deal_value
            FROM deals
            WHERE deal_value IS NOT NULL
        """,
        required_keywords=("average",),
        disallowed_keywords=("win",),
    ),
    # 17. Average won deal size
    QueryTemplate(
        id="average_won_deal_size",
        intent="Average deal size of won opportunities",
        keywords=("average", "win", "won", "deal", "size", "value", "deals", "opportunities"),
        sql="""
            SELECT 
                COUNT(*) AS total_won_deals,
                COUNT(deal_value) AS valued_won_deals_count,
                AVG(deal_value) AS avg_won_deal_value,
                SUM(deal_value) AS total_won_deal_value
            FROM deals
            WHERE deal_status = 'Won' AND deal_value IS NOT NULL
        """,
        required_keywords=("average", "win"),
    ),
    # 18. Dead deals by recorded loss reason/stage
    QueryTemplate(
        id="dead_deals_by_reason",
        intent="Dead deals grouped by recorded loss reason/stage",
        keywords=("dead", "reason", "loss", "stage", "lost", "deals", "potential", "revenue"),
        sql="""
            SELECT 
                COALESCE(deal_stage, 'Unknown') AS deal_stage,
                COUNT(*) AS dead_deal_count,
                COUNT(deal_value) AS valued_dead_count,
                SUM(deal_value) AS total_lost_deal_value
            FROM deals
            WHERE deal_status = 'Dead'
            GROUP BY deal_stage
            ORDER BY dead_deal_count DESC
        """,
        required_keywords=("dead",),
    ),
    # 19. Work order counts by sector
    QueryTemplate(
        id="work_orders_by_sector",
        intent="Work order count and value by sector",
        keywords=("work_order", "sector", "count", "industry", "vertical", "contracted"),
        sql="""
            SELECT 
                COALESCE(sector, 'Unknown') AS sector,
                COUNT(*) AS work_order_count,
                SUM(amount_incl_gst) AS total_contracted_incl_gst,
                SUM(billed_incl_gst) AS total_billed_incl_gst,
                SUM(collected_incl_gst) AS total_collected_incl_gst
            FROM work_orders
            GROUP BY sector
            ORDER BY work_order_count DESC
        """,
        required_keywords=("work_order", "sector"),
    ),
    # 20. Work order counts by execution status
    QueryTemplate(
        id="work_orders_by_execution_status",
        intent="Work order counts by execution status",
        keywords=("work_order", "execution_status", "status", "count", "mix", "distribution"),
        sql="""
            SELECT 
                COALESCE(execution_status, 'Unknown') AS status,
                COUNT(*) AS work_order_count,
                ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM work_orders), 2) AS pct_of_total,
                SUM(amount_incl_gst) AS total_contracted_incl_gst,
                SUM(billed_incl_gst) AS total_billed_incl_gst
            FROM work_orders
            GROUP BY execution_status
            ORDER BY work_order_count DESC
        """,
        required_keywords=("work_order", "execution_status"),
    ),
    # 21. Contracted revenue by fiscal quarter
    QueryTemplate(
        id="contracted_revenue_by_fiscal_quarter",
        intent="Contracted work-order revenue by PO fiscal quarter",
        keywords=("contracted", "quarter", "revenue", "po", "work_order", "fiscal"),
        sql="""
            SELECT 
                COALESCE(po_date_fq, 'Unknown') AS po_fiscal_quarter,
                COUNT(*) AS work_orders_count,
                SUM(amount_incl_gst) AS total_contracted_incl_gst,
                SUM(billed_incl_gst) AS total_billed_incl_gst
            FROM work_orders
            GROUP BY po_date_fq
            ORDER BY po_fiscal_quarter
        """,
        required_keywords=("contracted", "quarter"),
    ),
    # 22. Oldest open deals (row detail: includes item_id)
    QueryTemplate(
        id="oldest_open_deals",
        intent="Oldest open opportunities currently in pipeline",
        keywords=("oldest", "longest", "earliest", "pipeline", "deals", "open", "age"),
        sql="""
            SELECT 
                item_id,
                deal_name,
                owner_code,
                client_code,
                sector,
                deal_stage,
                deal_value,
                created_date
            FROM deals
            WHERE deal_status = 'Open' AND created_date IS NOT NULL
            ORDER BY created_date ASC
            LIMIT 5
        """,
        required_keywords=("oldest", "pipeline"),
    ),
    # 23. Won deals by owner
    QueryTemplate(
        id="won_deals_by_owner",
        intent="Won deals volume and value by owner",
        keywords=("win", "won", "owner", "rep", "count", "brought", "value", "closed"),
        sql="""
            SELECT 
                owner_code,
                COUNT(*) AS won_deals_count,
                COUNT(deal_value) AS valued_won_count,
                SUM(deal_value) AS total_won_deal_value
            FROM deals
            WHERE deal_status = 'Won'
            GROUP BY owner_code
            ORDER BY won_deals_count DESC, total_won_deal_value DESC
        """,
        required_keywords=("win", "owner"),
    ),
    # 24. Open deals by stage
    QueryTemplate(
        id="open_deals_by_stage",
        intent="Open deals breakdown by stage",
        keywords=("pipeline", "stage", "stages", "deal_stage", "funnel", "breakdown"),
        sql="""
            SELECT 
                COALESCE(deal_stage, 'Unknown') AS deal_stage,
                deal_stage_order,
                COUNT(*) AS deal_count,
                COUNT(deal_value) AS valued_deal_count,
                SUM(deal_value) AS raw_pipeline_value,
                SUM(weighted_value) AS weighted_pipeline_value
            FROM deals
            WHERE deal_status = 'Open'
            GROUP BY deal_stage, deal_stage_order
            ORDER BY deal_stage_order ASC NULLS LAST, deal_count DESC
        """,
        required_keywords=("stage",),
    ),
    # 25. Deals on hold (row detail: includes item_id)
    QueryTemplate(
        id="deals_on_hold",
        intent="Deals currently on hold",
        keywords=("hold", "on_hold", "deals", "recorded", "value", "count"),
        sql="""
            SELECT 
                item_id,
                deal_name,
                owner_code,
                client_code,
                sector,
                deal_stage,
                deal_value
            FROM deals
            WHERE deal_status = 'On Hold'
            ORDER BY deal_value DESC NULLS LAST
        """,
        required_keywords=("hold",),
    ),
    # 26. Fully paid work orders summary
    QueryTemplate(
        id="fully_paid_work_orders_summary",
        intent="Fully paid work orders with zero outstanding balance",
        keywords=("fully_paid", "paid", "zero", "receivables", "balance", "work_order"),
        sql="""
            SELECT 
                COUNT(*) AS fully_paid_work_orders_count,
                SUM(amount_incl_gst) AS total_contracted_value,
                SUM(collected_incl_gst) AS total_collected_value
            FROM work_orders
            WHERE billed_incl_gst > 0 AND outstanding_incl_gst = 0
        """,
        required_keywords=("fully_paid",),
    ),
    # 27. Ongoing work orders (row detail: includes item_id)
    QueryTemplate(
        id="ongoing_work_orders_detail",
        intent="Ongoing work orders execution details",
        keywords=("ongoing", "work_order", "active", "execution", "value", "count"),
        sql="""
            SELECT 
                item_id,
                deal_name,
                customer_code,
                sector,
                amount_incl_gst,
                billed_incl_gst,
                collected_incl_gst,
                outstanding_incl_gst
            FROM work_orders
            WHERE execution_status = 'Ongoing'
            ORDER BY amount_incl_gst DESC NULLS LAST
        """,
        required_keywords=("ongoing", "work_order"),
    ),
    # 28. Top completed work orders (row detail: includes item_id)
    QueryTemplate(
        id="top_completed_work_orders",
        intent="Top largest completed work orders by contract value",
        keywords=("completed", "largest", "work_order", "contracted", "value", "top"),
        sql="""
            SELECT 
                item_id,
                deal_name,
                customer_code,
                sector,
                amount_incl_gst,
                billed_incl_gst,
                collected_incl_gst
            FROM work_orders
            WHERE execution_status = 'Completed'
            ORDER BY amount_incl_gst DESC NULLS LAST
            LIMIT 10
        """,
        required_keywords=("completed", "largest"),
    ),
    # 29. Total unbilled balance summary
    QueryTemplate(
        id="total_unbilled_summary",
        intent="Total unbilled amount across all work orders",
        keywords=("unbilled", "total", "work_order", "amount", "across", "balance"),
        sql="""
            SELECT 
                COUNT(*) AS total_work_orders,
                SUM(unbilled_incl_gst) AS total_unbilled_incl_gst,
                SUM(amount_incl_gst) AS total_contracted_incl_gst,
                SUM(billed_incl_gst) AS total_billed_incl_gst
            FROM work_orders
        """,
        required_keywords=("unbilled", "total"),
        disallowed_keywords=("sector",),
    ),
    # 30. Unbilled balance by sector
    QueryTemplate(
        id="unbilled_balance_by_sector",
        intent="Unbilled balance breakdown by sector",
        keywords=("unbilled", "sector", "work_order", "industry", "vertical", "breakdown"),
        sql="""
            SELECT 
                COALESCE(sector, 'Unknown') AS sector,
                COUNT(*) AS work_orders_count,
                SUM(unbilled_incl_gst) AS unbilled_balance,
                SUM(amount_incl_gst) AS contracted_amount
            FROM work_orders
            GROUP BY sector
            ORDER BY unbilled_balance DESC NULLS LAST
        """,
        required_keywords=("unbilled", "sector"),
    ),
    # 31. Cross-board matching open deal names
    QueryTemplate(
        id="cross_board_matching_open_deals",
        intent="Cross-board matching deal names for open pipeline and work orders",
        keywords=("cross_board", "matching", "match", "overlap", "pipeline", "work_order", "both"),
        sql="""
            WITH d AS (
                SELECT deal_name, SUM(deal_value) AS pipeline 
                FROM deals 
                WHERE deal_status = 'Open' AND deal_name IS NOT NULL 
                GROUP BY deal_name
            ), 
            w AS (
                SELECT deal_name, SUM(amount_incl_gst) AS wo_value 
                FROM work_orders 
                WHERE deal_name IS NOT NULL 
                GROUP BY deal_name
            ) 
            SELECT 
                COUNT(d.deal_name) AS matching_deal_names_count,
                SUM(d.pipeline) AS total_open_pipeline_value,
                SUM(w.wo_value) AS total_contracted_work_order_value 
            FROM d JOIN w USING (deal_name)
        """,
        required_keywords=("matching",),
    ),
    # 32. Cross-board total shared distinct deal names
    QueryTemplate(
        id="cross_board_shared_distinct_names",
        intent="Count distinct deal names shared across both boards",
        keywords=("distinct", "shared", "names", "appear", "both", "intersection", "overlap"),
        sql="""
            WITH d AS (SELECT DISTINCT deal_name FROM deals WHERE deal_name IS NOT NULL),
                 w AS (SELECT DISTINCT deal_name FROM work_orders WHERE deal_name IS NOT NULL)
            SELECT COUNT(*) AS shared_distinct_deal_names
            FROM d JOIN w USING (deal_name)
        """,
        required_keywords=("distinct", "shared"),
    ),
    # 33. Total weighted pipeline
    QueryTemplate(
        id="total_weighted_pipeline",
        intent="Total probability-weighted pipeline for all open deals",
        keywords=("weighted", "probability", "pipeline", "total", "open", "all"),
        sql="""
            SELECT 
                COUNT(*) AS total_open_deals,
                COUNT(weighted_value) AS weighted_deals_count,
                SUM(weighted_value) AS total_weighted_pipeline_value,
                SUM(deal_value) AS total_raw_pipeline_value
            FROM deals 
            WHERE deal_status = 'Open'
        """,
        required_keywords=("weighted",),
    ),
    # 34. Total won deal value overall
    QueryTemplate(
        id="total_won_deal_value_overall",
        intent="Total won deal value across all owners",
        keywords=("win", "won", "total", "value", "deals", "revenue", "overall", "owners"),
        sql="""
            SELECT 
                COUNT(*) AS total_won_deals,
                COUNT(deal_value) AS valued_won_deals_count,
                SUM(deal_value) AS total_won_deal_value
            FROM deals 
            WHERE deal_status = 'Won'
        """,
        required_keywords=("win", "total"),
        disallowed_keywords=("sector", "owner"),
    ),
    # 35. Total lost revenue from dead deals
    QueryTemplate(
        id="total_dead_deals_lost_value",
        intent="Total potential revenue lost in dead deals",
        keywords=("dead", "lost", "potential", "revenue", "value", "deals", "total"),
        sql="""
            SELECT 
                COUNT(*) AS total_dead_deals,
                COUNT(deal_value) AS valued_dead_deals_count,
                SUM(deal_value) AS total_lost_deal_value
            FROM deals 
            WHERE deal_status = 'Dead'
        """,
        required_keywords=("dead", "revenue"),
    ),
    # 36. Renewables sector pipeline
    QueryTemplate(
        id="renewables_pipeline_summary",
        intent="Renewables sector open pipeline summary",
        keywords=("renewables", "pipeline", "sector", "open", "value"),
        sql="""
            SELECT 
                COUNT(*) AS open_deal_count,
                COUNT(deal_value) AS valued_deal_count,
                SUM(deal_value) AS raw_open_pipeline_value,
                SUM(weighted_value) AS weighted_pipeline_value
            FROM deals 
            WHERE deal_status = 'Open' AND sector ILIKE '%Renewables%'
        """,
        required_keywords=("renewables",),
    ),
    # 37. Mining sector pipeline
    QueryTemplate(
        id="mining_pipeline_summary",
        intent="Mining sector open pipeline summary",
        keywords=("mining", "pipeline", "sector", "open", "value"),
        sql="""
            SELECT 
                COUNT(*) AS open_deal_count,
                COUNT(deal_value) AS valued_deal_count,
                SUM(deal_value) AS raw_open_pipeline_value,
                SUM(weighted_value) AS weighted_pipeline_value
            FROM deals 
            WHERE deal_status = 'Open' AND sector ILIKE '%Mining%'
        """,
        required_keywords=("mining", "pipeline"),
    ),
    # 38. Railways sector pipeline
    QueryTemplate(
        id="railways_pipeline_summary",
        intent="Railways sector open pipeline summary",
        keywords=("railways", "pipeline", "sector", "open", "value"),
        sql="""
            SELECT 
                COUNT(*) AS open_deal_count,
                COUNT(deal_value) AS valued_deal_count,
                SUM(deal_value) AS raw_open_pipeline_value,
                SUM(weighted_value) AS weighted_pipeline_value
            FROM deals 
            WHERE deal_status = 'Open' AND sector ILIKE '%Railways%'
        """,
        required_keywords=("railways", "pipeline"),
    ),
    # 39. Powerline sector pipeline
    QueryTemplate(
        id="powerline_pipeline_summary",
        intent="Powerline sector open pipeline summary",
        keywords=("powerline", "pipeline", "sector", "open", "value"),
        sql="""
            SELECT 
                COUNT(*) AS open_deal_count,
                COUNT(deal_value) AS valued_deal_count,
                SUM(deal_value) AS raw_open_pipeline_value,
                SUM(weighted_value) AS weighted_pipeline_value
            FROM deals 
            WHERE deal_status = 'Open' AND sector ILIKE '%Powerline%'
        """,
        required_keywords=("powerline", "pipeline"),
    ),
    # 40. Mining sector receivables summary
    QueryTemplate(
        id="mining_receivables_summary",
        intent="Mining sector outstanding receivables summary",
        keywords=("mining", "receivables", "sector", "owed", "outstanding"),
        sql="""
            SELECT 
                COUNT(*) AS work_order_count,
                SUM(billed_incl_gst) AS total_billed_incl_gst,
                SUM(collected_incl_gst) AS total_collected_incl_gst,
                SUM(outstanding_incl_gst) AS total_outstanding_receivable
            FROM work_orders
            WHERE sector ILIKE '%Mining%'
        """,
        required_keywords=("mining", "receivables"),
    ),
]

# --------------------------------------------------------------------------- #
# Pure-Python Tokenizer & Scorer
# --------------------------------------------------------------------------- #

# Justification for threshold:
# 0.44 is deliberately just above the highest observed false-positive score
# (0.438 for an owner-ranking question incorrectly matching the largest
# individual open deal; an earlier age/staleness false positive scored 0.400).
# The deterministic floor should prefer an honest unsupported response to a
# plausible-but-wrong SQL answer during a provider outage.
TEMPLATE_MATCH_THRESHOLD: float = 0.44


def tokenize(text: str) -> list[str]:
    """Tokenize and canonicalize query text using word boundary normalization and synonyms."""
    t = text.lower()
    # Strip contractions and possessives
    t = re.sub(r"['’]s\b", " ", t)
    t = re.sub(r"n['’]t\b", " not ", t)

    # Standardize common domain multi-word phrases into distinct tokens
    t = re.sub(r"\bwin\s+rates?\b", " win_rate ", t)
    t = re.sub(r"\bwin\s+percentages?\b", " win_rate ", t)
    t = re.sub(r"\bwork\s+orders?\b", " work_order ", t)
    t = re.sub(r"\bopen\s+deals?\b", " pipeline ", t)
    t = re.sub(r"\bopen\s+pipeline\b", " pipeline ", t)
    t = re.sub(r"\bdeal\s+owners?\b", " owner ", t)
    t = re.sub(r"\bdeal\s+status\b", " deal_status ", t)
    t = re.sub(r"\bexecution\s+status\b", " execution_status ", t)
    t = re.sub(r"\bdeal\s+stages?\b", " deal_stage ", t)
    t = re.sub(r"\bfiscal\s+quarters?\b", " quarter ", t)
    t = re.sub(r"\bnot\s+billed\b", " unbilled ", t)
    t = re.sub(r"\bnot\s+invoiced\b", " unbilled ", t)
    t = re.sub(r"\bzero\s+billing\b", " unbilled ", t)
    t = re.sub(r"\bzero\s+billed\b", " unbilled ", t)
    t = re.sub(r"\bfully\s+paid\b", " fully_paid ", t)
    t = re.sub(r"\bclosed\s+outcomes?\b", " win dead ", t)
    t = re.sub(r"\bloss\s+reason\b", " dead reason ", t)
    t = re.sub(r"\bzero\s+invoice(?:\s+amount)?\b", " unbilled ", t)
    t = re.sub(r"\bprobability\s+weighted\b", " weighted ", t)
    t = re.sub(r"\bweighted\s+pipeline\b", " weighted ", t)
    t = re.sub(r"\braw\s+pipeline\b", " pipeline raw ", t)
    t = re.sub(r"\bshared\s+by\s+both\b", " distinct shared ", t)
    t = re.sub(r"\bboth\s+boards\b", " distinct shared ", t)
    t = re.sub(r"\bmatching\s+work\s+orders\b", " matching work_order ", t)

    # Strip non-alphanumeric characters
    t = re.sub(r"[^\w\s]", " ", t)
    words = t.split()
    tokens: list[str] = []
    for w in words:
        if w in STOPWORDS or len(w) <= 1:
            continue
        mapped = SYNONYMS.get(w, w)
        tokens.append(mapped)
    return tokens


def score_template(tokens: list[str], template: QueryTemplate) -> float:
    """Calculate overlap score between query tokens and template keywords."""
    if not tokens:
        return 0.0
    token_set = set(tokens)

    # Required and disallowed keyword constraints
    if template.required_keywords and not all(rk in token_set for rk in template.required_keywords):
        return 0.0
    if template.disallowed_keywords and any(dk in token_set for dk in template.disallowed_keywords):
        return 0.0

    kw_set = set(template.keywords)
    overlap = token_set & kw_set
    if not overlap:
        return 0.0

    # Query coverage: fraction of the user's meaningful query tokens matched
    query_cov = len(overlap) / len(token_set)
    # Template recall: fraction of template's distinguishing keywords matched
    template_rec = len(overlap) / len(kw_set)

    # Weight query coverage heavily (75%) with template recall for tie-breaking (25%)
    return 0.75 * query_cov + 0.25 * template_rec


def match_template(question: str) -> tuple[QueryTemplate | None, float]:
    """Score all templates against query and return (best_template, score)."""
    tokens = tokenize(question)
    if not tokens:
        return None, 0.0

    scored: list[tuple[float, QueryTemplate]] = []
    for template in TEMPLATES:
        score = score_template(tokens, template)
        if score > 0.0:
            scored.append((score, template))

    if not scored:
        return None, 0.0

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_template = scored[0]
    return best_template, best_score
