select
	sum(case when source = 'legislation.govt.nz' then 1 else 0 end) as acts_from_legislation_govt_nz_subscribe,
	sum(case when source = 'www.legislation.govt.nz' then 1 else 0 end) as acts_from_legislation_govt_nz_html,
	sum(case when source = 'nzlii.org' then 1 else 0 end) as acts_from_nzlii_org,
	sum(case when source is not null then 1 else 0 end) as total_acts,
	sum(case when text is null then 1 else 0 end) as acts_no_text_available,
	max(word_count) as max_word_count,
	min(word_count) as min_word_count,
	avg(word_count) as mean_word_count,
	sum(word_count) as total_word_count,
	percentile_cont(0.5) within group (order by word_count) as median_word_count,
	percentile_cont(0.6) within group (order by word_count) as p60_word_count,
	percentile_cont(0.7) within group (order by word_count) as p70_word_count,
	percentile_cont(0.8) within group (order by word_count) as p80_word_count,
	percentile_cont(0.9) within group (order by word_count) as p90_word_count,
	percentile_cont(0.95) within group (order by word_count) as p95_word_count,
	percentile_cont(0.98) within group (order by word_count) as p98_word_count,
	percentile_cont(0.99) within group (order by word_count) as p99_word_count,
	sum(case when word_count > 10000 then 1 else 0 end) as count_over_10000
from legislations
where (source in ('legislation.govt.nz', 'www.legislation.govt.nz', 'nzlii.org') or text is null)
	 or word_count > 0;


select
	title,
	pdf_link,
	word_count,
	(
		select count(*)
		from act_relationships
		where subject_name = legislations.title
	) as relationship_count
from legislations
where word_count < 9000
	and source = 'legislation.govt.nz' 
order by word_count desc
limit 100;

select count(*) from act_relationships where subject_name = 'Securities Amendment Act 2002';
