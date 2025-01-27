/*
 * althought this model is designed to show the cheapest shipper by region, there really aren't enough single product orders to make a determination
 * It might be possible to compare similar order or to weight the orders by the product(s) 
 */

with product_cnt as (
select
   *,
   count(*) over(partition by orderid) as product_cnt_in_order
from "Order Details" od
)
select
   pc.productid,
   s.CompanyName as freight_company,
   o.ShipRegion,
   sum(o.Freight)/sum(od.Quantity) as avg_freight_per_unit,
   ROW_NUMBER() over(partition by pc.productid, o.shipregion) as cheapest_shipper_for_region,
   count(*) as n -- this shows there really isn't enough single product orders to decide the cheapest shipper
from product_cnt pc
join {{ source('northwind', 'order_details') }} od on pc.orderid = od.orderid
join {{ source('northwind', 'orders') }} o on pc.orderid = o.orderid
join {{ source('northwind', 'shippers') }} s on o.shipvia = s.ShipperID
where product_cnt_in_order = 1
group by pc.productid, s.CompanyName, o.shipregion
order by 1, 3, 4
