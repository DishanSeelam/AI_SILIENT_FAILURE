# E-commerce dataset dictionary

`raw/ecommerce_transactions.csv` is a reproducible synthetic dataset with 2,000 orders. It is for project development only; it does not contain real customers or transactions.

| Column | Type | Description |
| --- | --- | --- |
| `transaction_id` | text | Unique order transaction identifier; exclude from model features. |
| `transaction_timestamp` | datetime | Time of purchase; derive temporal features before training. |
| `customer_id` | text | Synthetic customer identifier; exclude from model features. |
| `customer_age` | integer | Customer age in years. |
| `country` | category | Customer country. |
| `membership_tier` | category | Loyalty membership level. |
| `customer_tenure_days` | integer | Days since customer account creation. |
| `product_category` | category | Purchased product group. |
| `items_count` | integer | Number of items ordered. |
| `unit_price` | decimal | Price per item before discount. |
| `discount_percent` | integer | Discount applied to the order. |
| `order_value` | decimal | Total after discount. |
| `payment_method` | category | Payment channel. |
| `device_type` | category | Device used for purchase. |
| `shipping_distance_km` | decimal | Synthetic delivery distance. |
| `delivery_days_estimate` | integer | Estimated delivery time. |
| `is_weekend` | binary | 1 if the purchase was placed Saturday/Sunday. |
| `hour_of_day` | integer | Hour (0–23) in which it was placed. |
| `previous_orders` | integer | Prior completed orders by the customer. |
| `previous_chargebacks` | integer | Prior payment disputes. |
| `is_fraudulent` | binary target | 1 for a flagged fraudulent transaction; 0 otherwise. |

## Suggested model target

Train the first classifier using `is_fraudulent` as the target. The transaction and customer identifiers should not be model inputs. The timestamp should be converted into useful features such as month, day of week, and hour.

## Intended monitoring tests

- Change the distribution of `order_value` or `customer_age` to demonstrate data drift.
- Add missing values or invalid categories to demonstrate feature validation.
- Send unusual high-value, overnight orders to demonstrate prediction anomalies.
- Change encoding or preprocessing to demonstrate training-serving skew.
