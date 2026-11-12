import unittest
from decimal import Decimal, localcontext
from itertools import product

from bot.expense_split import (Allocation, add_participant, assign_item, complete, discount_shares,
                               participant_totals, restaurant_shares, snapshot)
from bot.settlements import (balances, confirm_settlement, expense_shares, set_discount, set_payment,
                             set_tips, settlement_issues, transfers)


class DiscountTests(unittest.TestCase):
    def plan(self, amounts=('1000','2000','1600'), *, gross='before', **facts):
        total = sum(map(Decimal,amounts))
        plan = snapshot({'items':[{'name':'Тестовая позиция','quantity':'1','total_price':value} for value in amounts],
                         'gross_total':str(total) if gross=='before' else gross, 'currency':'RUB', **facts},
                        user_id=42,chat_id=42)
        for index in range(3):plan=add_participant(plan,f'Участник {index+1}')
        for index,item in enumerate(plan.items):plan=assign_item(plan,item.id,[f'p{index%3}'])
        return complete(plan)

    def assert_reconciles(self,plan):
        net=balances(plan)
        self.assertEqual(sum(net.values()),0)
        for move in transfers(plan):
            net[move.sender_id]+=Decimal(move.amount)
            net[move.recipient_id]-=Decimal(move.amount)
        self.assertTrue(all(value==0 for value in net.values()))

    def test_example_4600_less_920(self):
        original=self.plan()
        plan=set_payment(set_discount(original,'920'),'p1','3680')
        self.assertEqual(discount_shares(plan),{'p0':Decimal('200'),'p1':Decimal('400'),'p2':Decimal('320')})
        self.assertEqual(restaurant_shares(plan),{'p0':Decimal('800'),'p1':Decimal('1600'),'p2':Decimal('1280')})
        self.assertEqual(plan.items_total,Decimal('4600'))
        self.assertEqual(plan.restaurant_total,Decimal('3680'))
        self.assertEqual(plan.expenses_total,Decimal('3680'))
        self.assertEqual([(m.sender_id,m.recipient_id,m.amount) for m in transfers(plan)],
                         [('p0','p1','800.00'),('p2','p1','1280.00')])
        for field in ('items','gross_total','receipt_total','prepayment','amount_due','currency','assignments'):
            self.assertEqual(plan.model_dump()[field],original.model_dump()[field])
        self.assertTrue(confirm_settlement(plan).settlement_confirmed)
        self.assert_reconciles(plan)

    def test_example_with_300_tips_keeps_balances_zero(self):
        plan=set_tips(set_payment(set_discount(self.plan(),'920'),'p1','3680'),'300','p2')
        self.assertEqual(expense_shares(plan),{'p0':Decimal('900'),'p1':Decimal('1700'),'p2':Decimal('1380')})
        self.assertEqual(balances(plan),{'p0':Decimal('-900'),'p1':Decimal('1980'),'p2':Decimal('-1080')})
        self.assertEqual([(m.sender_id,m.recipient_id,m.amount) for m in transfers(plan)],
                         [('p0','p1','900.00'),('p2','p1','1080.00')])
        self.assertEqual(plan.expenses_total,Decimal('3980'))
        self.assertEqual(plan.paid_total,Decimal('3680'))
        self.assert_reconciles(plan)

    def test_one_kopeck_equal_tie_uses_participant_order(self):
        plan=set_discount(self.plan(('1','1','1')),'0.01')
        self.assertEqual(discount_shares(plan),{'p0':Decimal('.01'),'p1':Decimal(0),'p2':Decimal(0)})
        self.assertEqual(restaurant_shares(plan),{'p0':Decimal('.99'),'p1':Decimal('1'),'p2':Decimal('1')})

    def test_largest_fractional_remainder_not_largest_position_or_first_payer(self):
        plan=set_discount(self.plan(('1','2','3')),'0.01')
        self.assertEqual(discount_shares(plan),{'p0':Decimal(0),'p1':Decimal(0),'p2':Decimal('.01')})
        tied=set_discount(self.plan(('0.01','0.01','0.04')),'0.02')
        self.assertEqual(discount_shares(tied),{'p0':Decimal('.01'),'p1':Decimal(0),'p2':Decimal('.01')})

    def test_hundred_percent_discount_zero_cost_and_separate_tips(self):
        plan=set_discount(self.plan(),'4600')
        self.assertTrue(all(value==0 for value in restaurant_shares(plan).values()))
        self.assertEqual(plan.restaurant_total,0)
        self.assertEqual(transfers(plan),[])
        self.assertTrue(confirm_settlement(plan).settlement_confirmed)
        plan=set_tips(plan,'300','p2')
        self.assertEqual(expense_shares(plan),dict.fromkeys(('p0','p1','p2'),Decimal('100')))
        self.assertEqual(plan.expenses_total,Decimal('300'))
        self.assert_reconciles(plan)

    def test_zero_share_receives_no_discount_but_still_shares_tips(self):
        plan=set_discount(self.plan(('1','2','0')),'0.02')
        self.assertEqual(discount_shares(plan)['p2'],0)
        self.assertEqual(restaurant_shares(plan)['p2'],0)
        plan=set_tips(set_payment(plan,'p0','2.98'),'0.03','p2')
        self.assertEqual(expense_shares(plan)['p2'],Decimal('.01'))
        self.assert_reconciles(plan)

    def test_shared_duplicate_items_and_click_order_do_not_change_discount(self):
        base=self.plan(('0.05','0.05'))
        for owners in (['p1','p0'],['p0','p1']):
            opened=Allocation.model_validate({**base.model_dump(),'completed':False})
            for item in opened.items:opened=assign_item(opened,item.id,owners)
            plan=set_discount(complete(opened),'0.03')
            self.assertEqual(participant_totals(plan),{'p0':Decimal('.06'),'p1':Decimal('.04'),'p2':Decimal(0)})
            self.assertEqual(discount_shares(plan),{'p0':Decimal('.02'),'p1':Decimal('.01'),'p2':Decimal(0)})
            self.assertEqual(restaurant_shares(plan),{'p0':Decimal('.04'),'p1':Decimal('.03'),'p2':Decimal(0)})
            self.assertEqual(len(plan.items),2)

    def test_change_delete_and_recalculation_never_accumulate(self):
        confirmed=confirm_settlement(set_payment(set_discount(self.plan(),'920'),'p1','3680'))
        changed=set_discount(confirmed,'460')
        self.assertFalse(changed.settlement_confirmed)
        self.assertEqual(changed.restaurant_total,Decimal('4140'))
        self.assertEqual(set_discount(changed,'460'),changed)
        self.assertEqual(discount_shares(changed),discount_shares(changed))
        self.assertTrue(confirmed.settlement_confirmed)
        removed=set_discount(changed,'0')
        self.assertEqual(restaurant_shares(removed),participant_totals(removed))
        self.assertEqual(removed.expenses_total,Decimal('4600'))
        self.assertFalse(removed.settlement_confirmed)
        self.assertIn('payments_mismatch',settlement_issues(removed))

    def test_invalid_values_and_schema_cannot_create_negative_expenses(self):
        plan=self.plan()
        for value in ('-1','0.001','1e2','NaN','Infinity','','4600.01','1000000000000000',0.5,True):
            with self.subTest(value=value),self.assertRaises(ValueError):set_discount(plan,value)
        for value in ('-1','0.001','4600.01',0.5,True,None):
            with self.subTest(value=value),self.assertRaises(ValueError):
                Allocation.model_validate({**plan.model_dump(),'discount_amount':value})
        unfinished=Allocation.model_validate({**plan.model_dump(),'completed':False})
        with self.assertRaises(ValueError):set_discount(unfinished,'1')
        self.assertEqual(set_discount(plan,' 0920,00 ').discount_amount,'920.00')

    def test_prepaid_amount_is_only_a_payment_not_an_extra_expense(self):
        original=self.plan(gross='3680',prepayment='1000',amount_due='2680')
        plan=set_discount(original,'920')
        self.assertEqual(plan.paid_total,0)
        plan=set_payment(plan,'p0','1000')
        self.assertIn('payments_mismatch',settlement_issues(plan))
        plan=set_payment(plan,'p1','2680')
        self.assertEqual(plan.expenses_total,Decimal('3680'))
        self.assertFalse(plan.mismatch)
        self.assertEqual((plan.gross_total,plan.prepayment,plan.amount_due),('3680','1000','2680'))
        self.assert_reconciles(plan)

    def test_gross_before_or_after_requires_explicit_discount_and_is_not_rewritten(self):
        for gross in ('4600','3680',None):
            original=self.plan(gross=gross,receipt_total=gross)
            plan=set_discount(original,'920')
            self.assertFalse(plan.mismatch)
            self.assertEqual(plan.gross_total,gross)
            self.assertEqual(plan.receipt_total,gross)
            self.assertTrue(confirm_settlement(set_payment(plan,'p1','3680')).settlement_confirmed)
        no_discount=set_payment(self.plan(gross='3680'),'p1','3680')
        self.assertEqual(no_discount.discount_amount,'0.00')
        with self.assertRaises(ValueError):confirm_settlement(no_discount)
        removed=set_discount(set_discount(no_discount,'920'),'0')
        self.assertTrue(removed.mismatch)

    def test_ambiguous_or_conflicting_ocr_totals_block_confirmation(self):
        cases=[{'gross':'4000'}, {'gross':'4600','receipt_total':'4000'},
               {'gross':'4000','receipt_total':'3680'}, {'gross':'3680','receipt_total':'4600'},
               {'gross':'3680','receipt_total':'4000'}, {'prepayment':'1000','amount_due':'3000'},
               {'amount_due':'2680'}, {'prepayment':'4000'}, {'prepayment_suspect':True}]
        for facts in cases:
            with self.subTest(facts=facts):
                original=self.plan(**facts)
                plan=set_payment(set_discount(original,'920'),'p1','3680')
                self.assertTrue(plan.mismatch)
                self.assertIn('receipt_mismatch',settlement_issues(plan))
                for action in (transfers,confirm_settlement):
                    with self.assertRaises(ValueError):action(plan)
                with self.assertRaises(ValueError):
                    Allocation.model_validate({**plan.model_dump(),'settlement_confirmed':True})
                self.assertEqual(plan.gross_total,original.gross_total)

    def test_after_discount_due_can_be_checked_against_before_discount_gross(self):
        for facts in ({'amount_due':'3680'},{'prepayment':'1000','amount_due':'2680'}):
            plan=set_payment(set_discount(self.plan(**facts),'920'),'p1','3680')
            self.assertFalse(plan.mismatch)
            self.assert_reconciles(plan)

    def test_before_and_after_totals_with_user_discount_can_be_confirmed(self):
        original=self.plan(gross='4600',receipt_total='3680')
        paid=set_payment(original,'p1','3680')
        self.assertEqual(paid.discount_amount,'0.00')
        self.assertEqual(paid.restaurant_total,Decimal('4600'))
        with self.assertRaises(ValueError):confirm_settlement(paid)
        discounted=set_discount(paid,'920')
        self.assertEqual(discounted.receipt_total_issues,[])
        self.assertEqual(discounted.paid_total,Decimal('3680'))
        self.assertEqual(discounted.restaurant_total,Decimal('3680'))
        self.assertEqual(settlement_issues(discounted),[])
        self.assertEqual([(m.sender_id,m.recipient_id,m.amount) for m in transfers(discounted)],
                         [('p0','p1','800.00'),('p2','p1','1280.00')])
        self.assertTrue(confirm_settlement(discounted).settlement_confirmed)
        for field in ('items','gross_total','receipt_total','prepayment','amount_due','currency'):
            self.assertEqual(discounted.model_dump()[field],original.model_dump()[field])
        self.assert_reconciles(discounted)

    def test_agreed_pair_does_not_bypass_wrong_discount_payments_or_suspect_prepayment(self):
        original=self.plan(gross='4600',receipt_total='3680')
        wrong=set_payment(set_discount(original,'919'),'p1','3681')
        self.assertIn('receipt_mismatch',settlement_issues(wrong))
        with self.assertRaises(ValueError):confirm_settlement(wrong)
        correct=set_discount(original,'920')
        for amount in ('3679.99','3680.01'):
            with self.subTest(amount=amount):
                wrong=set_payment(correct,'p1',amount)
                self.assertIn('payments_mismatch',settlement_issues(wrong))
                with self.assertRaises(ValueError):confirm_settlement(wrong)
        suspect=Allocation.model_validate({**correct.model_dump(),'prepayment_suspect':True,
                                            'payments':{'p1':'3680.00'}})
        self.assertIn('suspect_prepayment',suspect.receipt_total_issues)
        with self.assertRaises(ValueError):confirm_settlement(suspect)

    def test_legacy_zero_discount_loading_keeps_previous_reconciliation(self):
        original=set_payment(self.plan(),'p0','4600')
        legacy=Allocation.model_validate(original.model_dump(exclude={'discount_amount','receipt_total','prepayment_suspect'}))
        self.assertEqual(legacy.discount_amount,'0.00')
        self.assertEqual(balances(legacy),balances(original))
        self.assertTrue(confirm_settlement(legacy).settlement_confirmed)
        zero=set_discount(original,'0')
        self.assertEqual(transfers(zero),transfers(original))
        self.assertEqual(restaurant_shares(zero),participant_totals(original))

    def test_small_amounts_conserve_money_and_never_over_discount_a_person(self):
        for weights in product(range(4),repeat=3):
            plan=self.plan(tuple(str(Decimal(w)/100) for w in weights))
            before=participant_totals(plan)
            for cents in range(sum(weights)+1):
                changed=set_discount(plan,str(Decimal(cents)/100))
                discounts=discount_shares(changed)
                after=restaurant_shares(changed)
                self.assertEqual(sum(discounts.values()),Decimal(cents)/100)
                self.assertEqual(sum(after.values()),changed.restaurant_total)
                self.assertTrue(all(0<=discounts[p]<=before[p] for p in discounts))
                self.assertTrue(all(value>=0 for value in after.values()))

    def test_large_integer_remainders_are_exact_under_low_decimal_context(self):
        plan=self.plan(('999999999999.99',)*200,gross=None)
        with localcontext() as context:
            context.prec=2
            changed=set_discount(plan,'123456789012345.67')
            discounts=discount_shares(changed)
            after=restaurant_shares(changed)
            total=changed.restaurant_total
            paid=set_payment(changed,'p1',format(total,'.2f'))
            moves=transfers(paid)
        self.assertEqual(sum(discounts.values()),Decimal('123456789012345.67'))
        self.assertEqual(sum(after.values()),total)
        self.assertTrue(all(value>=0 for value in after.values()))
        self.assertTrue(moves)
        self.assert_reconciles(paid)

    def test_unassigned_discount_cannot_silently_lose_a_share(self):
        plan=set_discount(self.plan(),'920')
        opened=Allocation.model_validate({**plan.model_dump(),'completed':False})
        opened=assign_item(opened,'i0',[])
        with self.assertRaises(ValueError):discount_shares(opened)
        self.assertIn('allocation_incomplete',settlement_issues(opened))
