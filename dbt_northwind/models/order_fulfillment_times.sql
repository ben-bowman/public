with order_times as (
    select
        o.orderid,
        o.orderdate,
        o.shippeddate,
        julianday(o.shippeddate) - julianday(o.orderdate) as fulfillment_days
    from {{ source('northwind', 'orders') }} o
    where o.shippeddate is not null
), stats as (
SELECT
   avg(fulfillment_days) as avg_fulfillment_days
from order_times
), variances as (
select
  ot.orderid,
  s.avg_fulfillment_days,
  (ot.fulfillment_days - s.avg_fulfillment_days)*(ot.fulfillment_days - s.avg_fulfillment_days) as variance
from order_times ot
cross join stats s
) --outliers
select
    ot.*,
    case
      when ot.fulfillment_days > (v.avg_fulfillment_days + 2 * sqrt(v.variance)) then 'Long Fulfillment'
      else 'Normal Fulfillment'
      end as fulfillment_flag
from variances v
join order_times ot on v.orderid = ot.orderid

