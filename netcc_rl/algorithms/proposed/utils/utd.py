"""Exact transition-denominated update credit, independent of collector cadence."""
from fractions import Fraction
import math


class UpdateBudget:
    def __init__(self,utd_ratio):
        if isinstance(utd_ratio,bool): raise ValueError('utd_ratio must be positive and finite')
        try: self.ratio=Fraction(str(utd_ratio))
        except (ValueError,ZeroDivisionError): raise ValueError('utd_ratio must be positive and finite') from None
        if self.ratio<=0: raise ValueError('utd_ratio must be positive and finite')
        self.credit=Fraction(0)
        self.collected=self.enabled_transitions=self.updates=0

    def collect(self,newly_collected_transitions,*,enabled):
        if type(newly_collected_transitions) is not int or newly_collected_transitions<0:
            raise ValueError('actual transition count must be a nonnegative integer')
        if type(enabled) is not bool: raise ValueError('enabled must be boolean')
        self.collected+=newly_collected_transitions
        if enabled:
            self.enabled_transitions+=newly_collected_transitions
            self.credit+=newly_collected_transitions*self.ratio
        return self.credit.numerator//self.credit.denominator

    def updated(self):
        if self.credit<1: raise RuntimeError('no learner update credit')
        self.credit-=1
        self.updates+=1

    def diagnostics(self):
        return dict(collected_transitions=self.collected,utd_enabled_transitions=self.enabled_transitions,
                    learner_updates=self.updates,update_credit=float(self.credit),
                    update_credit_numerator=self.credit.numerator,update_credit_denominator=self.credit.denominator,
                    utd_ratio=float(self.ratio),effective_utd=self.updates/self.enabled_transitions if self.enabled_transitions else 0.,
                    effective_utd_all_collected=self.updates/self.collected if self.collected else 0.)


def validate_schedule(cfg):
    ratio=getattr(cfg,'utd_ratio',None)
    if ratio is None: return
    UpdateBudget(ratio)
    if cfg.updates_per_step is not None or cfg.initial_updates!=0:
        raise ValueError('utd_ratio requires updates_per_step=null and initial_updates=0')
